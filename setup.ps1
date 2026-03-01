# init.ps1

# INSTRUCTIONS:
# 1. Open TERMINAL and navigate to the root directory of your project.
# 2. Run this script: .\setup.ps1
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
Write-Output "✅ Structure created!"


