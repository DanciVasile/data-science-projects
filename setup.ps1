# init.ps1

# INSTRUCTIONS:
# 1. Open TERMINAL and navigate to the root directory of your project.
# 2. Run this script: & ..\setup.ps1
# 3. This will create the starting project template structure:

mkdir data
mkdir docs
mkdir models
mkdir notebooks
mkdir reports/figures
mkdir src
New-Item -Path data\.gitkeep -ItemType File -Force
New-Item -Path models\.gitkeep -ItemType File -Force
New-Item -Path reports/figures\.gitkeep -ItemType File -Force

# Generate the starter exploration notebook inside notebooks/
# Note: This assumes you have Python installed and available in your PATH.
# You can also cd into the folder you want to create it and run(if you have uv):
# uv run ..\create_notebook.py
# or the classic:
# python ..\create_notebook.py
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
python "$scriptDir\create_notebook.py" notebooks

Write-Output "✅ Structure created!"


