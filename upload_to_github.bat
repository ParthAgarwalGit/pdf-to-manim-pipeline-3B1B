@echo off
REM PDF to Manim Pipeline - GitHub Upload Script (Windows)
REM This script helps you upload the project to GitHub

echo.
echo ========================================
echo PDF to Manim Pipeline - GitHub Upload Helper
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo Error: Git is not installed
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

echo Git is installed
echo.

REM Check if already a git repository
if exist ".git" (
    echo Git repository already initialized
) else (
    echo Initializing Git repository...
    git init
    echo Git repository initialized
)

echo.
echo Please provide your GitHub repository details:
echo.

REM Get GitHub username
set /p github_username="Enter your GitHub username: "

REM Get repository name
set /p repo_name="Enter repository name [pdf-to-manim-pipeline]: "
if "%repo_name%"=="" set repo_name=pdf-to-manim-pipeline

REM Get visibility
echo.
echo Repository visibility:
echo 1) Public (anyone can see)
echo 2) Private (only you can see)
set /p visibility_choice="Choose (1 or 2) [1]: "
if "%visibility_choice%"=="" set visibility_choice=1

if "%visibility_choice%"=="2" (
    set visibility=private
) else (
    set visibility=public
)

echo.
echo Summary:
echo   Username: %github_username%
echo   Repository: %repo_name%
echo   Visibility: %visibility%
echo.

set /p confirm="Continue? (y/n) [y]: "
if "%confirm%"=="" set confirm=y

if /i not "%confirm%"=="y" (
    echo Cancelled
    pause
    exit /b 0
)

echo.
echo Setting up repository...

REM Add all files
echo Adding files...
git add .

REM Create initial commit
echo Creating initial commit...
git commit -m "Initial commit: PDF to Manim pipeline implementation"

REM Check if GitHub CLI is installed
gh --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo GitHub CLI not found
    echo.
    echo Manual steps:
    echo 1. Go to https://github.com/new
    echo 2. Create a new repository named: %repo_name%
    echo 3. Choose visibility: %visibility%
    echo 4. DO NOT initialize with README, .gitignore, or license
    echo 5. Click 'Create repository'
    echo.
    echo Then run these commands:
    echo.
    echo   git remote add origin https://github.com/%github_username%/%repo_name%.git
    echo   git branch -M main
    echo   git push -u origin main
    echo.
) else (
    echo.
    echo GitHub CLI detected
    echo Creating repository and pushing...
    
    if "%visibility%"=="private" (
        gh repo create %repo_name% --private --source=. --remote=origin --push
    ) else (
        gh repo create %repo_name% --public --source=. --remote=origin --push
    )
    
    if errorlevel 1 (
        echo.
        echo Failed to create repository with GitHub CLI
        echo Please create the repository manually on GitHub
    ) else (
        echo.
        echo Success! Repository created and code pushed!
        echo View at: https://github.com/%github_username%/%repo_name%
    )
)

echo.
echo Next steps:
echo 1. Update README.md with your information
echo 2. Add topics to your repository (python, manim, pdf, animation, llm)
echo 3. Set up GitHub Actions for automated testing (optional)
echo 4. Configure API keys for production use (see SETUP.md)
echo.
echo Done!
echo.
pause
