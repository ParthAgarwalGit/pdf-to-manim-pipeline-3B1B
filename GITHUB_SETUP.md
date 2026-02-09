# How to Upload This Project to GitHub

## Prerequisites

1. **GitHub Account**: Create one at https://github.com if you don't have one
2. **Git Installed**: Verify with `git --version`
3. **GitHub CLI (Optional)**: Install from https://cli.github.com/

## Method 1: Using GitHub Web Interface (Easiest)

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Fill in repository details:
   - **Repository name**: `pdf-to-manim-pipeline`
   - **Description**: `Agent-orchestrated system that transforms academic PDFs into 3Blue1Brown-style animated educational videos using Manim`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

### Step 2: Initialize Git and Push

Open your terminal in the project directory and run:

```bash
# Navigate to project directory
cd C:\Users\Parth\Documents\3B1B

# Initialize git repository (if not already initialized)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: PDF to Manim pipeline implementation"

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pdf-to-manim-pipeline.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Method 2: Using GitHub CLI (Recommended)

```bash
# Navigate to project directory
cd C:\Users\Parth\Documents\3B1B

# Login to GitHub (if not already logged in)
gh auth login

# Create repository and push in one command
gh repo create pdf-to-manim-pipeline --public --source=. --remote=origin --push

# Or for private repository
gh repo create pdf-to-manim-pipeline --private --source=. --remote=origin --push
```

## Method 3: Using GitHub Desktop

1. Download and install GitHub Desktop from https://desktop.github.com/
2. Open GitHub Desktop
3. Click "File" → "Add Local Repository"
4. Browse to `C:\Users\Parth\Documents\3B1B`
5. Click "Publish repository"
6. Choose repository name and visibility
7. Click "Publish Repository"

## Important Files to Include

### Create .gitignore

Before pushing, create a `.gitignore` file to exclude unnecessary files:

```bash
# .gitignore content (already created in project)
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.hypothesis/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
output/
*.log
.env
*.pdf

# Temporary files
*.tmp
*.bak
```

### Create LICENSE

Choose a license (e.g., MIT License):

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Repository Structure on GitHub

After uploading, your repository will look like:

```
pdf-to-manim-pipeline/
├── .github/
│   └── workflows/          # CI/CD workflows (optional)
├── .kiro/
│   ├── specs/             # Feature specifications
│   └── steering/          # AI assistant guidance
├── docs/                  # Documentation
├── src/                   # Source code
├── tests/                 # Test suite
├── .gitignore            # Git ignore rules
├── LICENSE               # License file
├── README.md             # Project overview
├── SETUP.md              # Setup instructions
├── GITHUB_SETUP.md       # This file
├── requirements.txt      # Python dependencies
└── pytest.ini           # Test configuration
```

## Setting Up GitHub Actions (Optional)

Create `.github/workflows/tests.yml` for automated testing:

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## After Uploading

### 1. Add Repository Description

On GitHub repository page:
- Click "⚙️ Settings"
- Add description: "Agent-orchestrated PDF to Manim animation pipeline"
- Add topics: `python`, `manim`, `pdf`, `animation`, `llm`, `ai`, `education`

### 2. Create README Badges

Add to top of README.md:

```markdown
# PDF to Manim Pipeline

[![Tests](https://github.com/YOUR_USERNAME/pdf-to-manim-pipeline/workflows/Tests/badge.svg)](https://github.com/YOUR_USERNAME/pdf-to-manim-pipeline/actions)
[![Coverage](https://codecov.io/gh/YOUR_USERNAME/pdf-to-manim-pipeline/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/pdf-to-manim-pipeline)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
```

### 3. Enable GitHub Pages (Optional)

For documentation hosting:
1. Go to repository Settings
2. Scroll to "Pages"
3. Select source: "Deploy from a branch"
4. Select branch: `main` and folder: `/docs`
5. Save

### 4. Set Up Branch Protection

1. Go to Settings → Branches
2. Add rule for `main` branch
3. Enable:
   - Require pull request reviews
   - Require status checks to pass
   - Require branches to be up to date

## Sharing Your Repository

Once uploaded, share your repository:

```
Repository URL: https://github.com/YOUR_USERNAME/pdf-to-manim-pipeline
Clone command: git clone https://github.com/YOUR_USERNAME/pdf-to-manim-pipeline.git
```

## Updating the Repository

After making changes:

```bash
# Check status
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push origin main
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:

```bash
# Use personal access token
# 1. Go to GitHub Settings → Developer settings → Personal access tokens
# 2. Generate new token with 'repo' scope
# 3. Use token as password when prompted

# Or configure Git credential helper
git config --global credential.helper store
```

### Large Files

If you have files > 100MB:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pdf"
git lfs track "*.mp4"

# Add .gitattributes
git add .gitattributes
git commit -m "Configure Git LFS"
```

### Removing Sensitive Data

If you accidentally committed API keys:

```bash
# Remove from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch .env" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push origin --force --all
```

## Next Steps

1. ✅ Upload to GitHub
2. 📝 Update README with your information
3. 🏷️ Add topics and description
4. 🔒 Set up branch protection
5. 🤖 Configure GitHub Actions
6. 📢 Share with community
7. ⭐ Get stars!

## Resources

- GitHub Docs: https://docs.github.com/
- Git Basics: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
- GitHub CLI: https://cli.github.com/manual/
- GitHub Actions: https://docs.github.com/en/actions
