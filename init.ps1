# init.ps1 — Run once after cloning the repository.
#
# What it does:
#   1. Creates .venv and installs all dependencies (via uv)
#   2. Registers the .venv as a Jupyter kernel so notebooks just work
#
# Usage:
#   git clone https://github.com/DanciVasile/data-science-projects.git
#   cd data-science-projects
#   .\init.ps1

Write-Output "🔧 Installing dependencies..."
uv sync

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Output "❌ .venv was not created. Make sure 'uv' is installed: https://docs.astral.sh/uv/"
    exit 1
}

Write-Output "📓 Registering Jupyter kernel..."
& $venvPython -m ipykernel install --user --name data-science-projects --display-name "Data Science Projects"

Write-Output ""
Write-Output "✅ All set! Open the repo in VS Code and select the 'Data Science Projects' kernel in any notebook."
