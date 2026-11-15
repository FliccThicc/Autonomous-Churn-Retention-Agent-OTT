$ErrorActionPreference = 'Stop'

$targets = @(
    'C:\Users\kdhru\.local\bin\uv.exe',
    'C:\Users\kdhru\.vscode\extensions\meta.pyrefly-1.4.1-win32-x64\bin\pyrefly.exe'
)

$missing = @()
foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        Unblock-File -LiteralPath $target
        Write-Host "Unblocked: $target"
    }
    else {
        $missing += $target
        Write-Warning "Missing: $target"
    }
}

$uvPath = $targets[0]
if (Test-Path -LiteralPath $uvPath -PathType Leaf) {
    $uvCommand = Get-Command $uvPath -ErrorAction SilentlyContinue
    if ($null -eq $uvCommand) {
        $env:Path = "$(Split-Path -Parent $uvPath);$env:Path"
    }

    try {
        $uvVersion = & $uvPath --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "uv exited with code $LASTEXITCODE"
        }
        Write-Host "uv validation passed: $uvVersion"
    }
    catch {
        Write-Error "uv exists but did not execute successfully: $($_.Exception.Message)"
        exit 1
    }
}
else {
    Write-Host 'uv is missing. Recreate the project environment with:'
    Write-Host '  .\.venv\Scripts\python.exe -m pip install uv'
    Write-Host '  .\.venv\Scripts\uv.exe venv .venv'
}

$pyreflyPath = $targets[1]
if (-not (Test-Path -LiteralPath $pyreflyPath -PathType Leaf)) {
    Write-Host 'pyrefly is missing. Reinstall the VS Code extension, or install its CLI in the project venv with:'
    Write-Host '  .\.venv\Scripts\python.exe -m pip install pyrefly'
}

if ($missing.Count -gt 0) {
    Write-Warning "$($missing.Count) target file(s) were missing; see the remediation instructions above."
    exit 2
}
