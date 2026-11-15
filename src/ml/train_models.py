"""Train the churn, survival, and explainability model bundle."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import shap
from lifelines import CoxPHFitter
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.calibration import CalibratedClassifierCV


SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "ott_subscribers_2026.parquet"
PERSONAS_PATH = ROOT / "data" / "processed" / "flagship_personas.json"
BUNDLE_PATH = ROOT / "data" / "processed" / "models" / "churn_bundle.joblib"
LOGGER = logging.getLogger(__name__)

TARGET = "churned_30d"
DURATION = "days_to_churn"
EXCLUDED_FEATURES = {
    "subscriber_id",
    "snapshot_date",
    "churn_probability",
    "primary_churn_driver",
    TARGET,
    DURATION,
}


def _humanize(name: str, profile: dict[str, Any]) -> str:
    """Create a concise, profile-aware explanation for a model feature."""
    label = name.replace("_", " ").capitalize()
    value = profile.get(name)
    if value is None:
        return label
    return f"{label}: {value}"


class ChurnExplainer:
    """Serializable inference wrapper for churn and survival diagnostics."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        classifier: CalibratedClassifierCV,
        tree_model: lgb.LGBMClassifier,
        cox_model: CoxPHFitter,
        feature_names: list[str],
    ) -> None:
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.tree_model = tree_model
        self.cox_model = cox_model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(tree_model)

    def _transform_profile(self, user_profile: dict[str, Any]) -> pd.DataFrame:
        missing = [name for name in self.feature_names if name not in user_profile]
        if missing:
            raise ValueError(f"user_profile is missing required features: {missing}")
        frame = pd.DataFrame([{name: user_profile[name] for name in self.feature_names}])
        return pd.DataFrame(
            self.preprocessor.transform(frame),
            columns=self.feature_names,
            index=frame.index,
        )

    def score_and_explain(self, user_profile: dict[str, Any]) -> dict[str, Any]:
        """Return calibrated churn, survival urgency, and SHAP explanations."""
        encoded = self._transform_profile(user_profile)
        churn_probability = float(self.classifier.predict_proba(encoded)[0, 1])
        expected_days = float(self.cox_model.predict_expectation(encoded).iloc[0])

        shap_values = self.explainer.shap_values(encoded)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[-1])[0]
        else:
            values_array = np.asarray(shap_values)
            values = values_array[0, :, 1] if values_array.ndim == 3 else values_array[0]

        ranked = sorted(
            zip(self.feature_names, values, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        risk = [
            {
                "feature": feature,
                "shap_value": float(value),
                "description": _humanize(feature, user_profile),
            }
            for feature, value in ranked
            if value > 0
        ][:3]
        anchors = [
            {
                "feature": feature,
                "shap_value": float(value),
                "description": _humanize(feature, user_profile),
            }
            for feature, value in reversed(ranked)
            if value < 0
        ][:2]
        return {
            "churn_probability": churn_probability,
            "expected_days_to_churn": expected_days,
            "top_risk_drivers": risk,
            "top_retention_anchors": anchors,
        }


def _build_preprocessor(frame: pl.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    feature_names = [name for name in frame.columns if name not in EXCLUDED_FEATURES]
    categorical = [
        name for name in feature_names if frame.schema[name] == pl.String
    ]
    numeric = [name for name in feature_names if name not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor, categorical, numeric


def train_models() -> tuple[ChurnExplainer, dict[str, float]]:
    """Train all model heads and persist the inference bundle."""
    frame = pl.read_parquet(DATA_PATH)
    preprocessor, categorical, numeric = _build_preprocessor(frame)
    feature_names = categorical + numeric
    features = frame.select(feature_names).to_pandas()
    target = frame.get_column(TARGET).to_numpy()

    train_indices, test_indices = train_test_split(
        np.arange(frame.height),
        test_size=0.2,
        stratify=target,
        random_state=SEED,
    )
    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]
    y_train = target[train_indices]
    y_test = target[test_indices]
    x_train_encoded = preprocessor.fit_transform(x_train)
    x_test_encoded = preprocessor.transform(x_test)
    x_train_encoded = pd.DataFrame(x_train_encoded, columns=feature_names)
    x_test_encoded = pd.DataFrame(x_test_encoded, columns=feature_names)

    base_model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    classifier = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    classifier.fit(x_train_encoded, y_train)
    probabilities = classifier.predict_proba(x_test_encoded)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
    }

    survival_frame = x_train_encoded.copy()
    survival_frame[DURATION] = frame.get_column(DURATION).to_numpy()[train_indices]
    survival_frame[TARGET] = y_train
    cox_model = CoxPHFitter(penalizer=0.1)
    cox_model.fit(survival_frame, duration_col=DURATION, event_col=TARGET)

    fitted_tree = classifier.calibrated_classifiers_[0].estimator
    if not isinstance(fitted_tree, lgb.LGBMClassifier):
        raise TypeError("Calibrated classifier did not retain an LGBMClassifier estimator")
    bundle = ChurnExplainer(
        preprocessor=preprocessor,
        classifier=classifier,
        tree_model=fitted_tree,
        cox_model=cox_model,
        feature_names=feature_names,
    )
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)
    return bundle, metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle, metrics = train_models()
    for metric, value in metrics.items():
        LOGGER.info("%s: %.6f", metric, value)
        print(f"{metric}: {value:.6f}")

    personas = json.loads(PERSONAS_PATH.read_text(encoding="utf-8"))
    for persona in personas:
        diagnostic = bundle.score_and_explain(persona)
        print(json.dumps({"subscriber_id": persona["subscriber_id"], **diagnostic}, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from src.ml.train_models import main as package_main

    package_main()
