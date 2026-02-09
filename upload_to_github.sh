#!/bin/bash

# PDF to Manim Pipeline - GitHub Upload Script
# This script helps you upload the project to GitHub

echo "🚀 PDF to Manim Pipeline - GitHub Upload Helper"
echo "================================================"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed"
    echo "Please install Git from https://git-scm.com/"
    exit 1
fi

echo "✅ Git is installed"
echo ""

# Check if already a git repository
if [ -d ".git" ]; then
    echo "📁 Git repository already initialized"
else
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
fi

echo ""
echo "📝 Please provide your GitHub repository details:"
echo ""

# Get GitHub username
read -p "Enter your GitHub username: " github_username

# Get repository name (with default)
read -p "Enter repository name [pdf-to-manim-pipeline]: " repo_name
repo_name=${repo_name:-pdf-to-manim-pipeline}

# Get visibility
echo ""
echo "Repository visibility:"
echo "1) Public (anyone can see)"
echo "2) Private (only you can see)"
read -p "Choose (1 or 2) [1]: " visibility_choice
visibility_choice=${visibility_choice:-1}

if [ "$visibility_choice" = "2" ]; then
    visibility="private"
else
    visibility="public"
fi

echo ""
echo "📋 Summary:"
echo "  Username: $github_username"
echo "  Repository: $repo_name"
echo "  Visibility: $visibility"
echo ""

read -p "Continue? (y/n) [y]: " confirm
confirm=${confirm:-y}

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "❌ Cancelled"
    exit 0
fi

echo ""
echo "🔧 Setting up repository..."

# Add all files
echo "📦 Adding files..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: PDF to Manim pipeline implementation

- Complete agent-oriented architecture with 8 specialized agents
- Pipeline orchestration with retry policies and logging
- Comprehensive test suite (358 tests, 93% coverage)
- Deterministic operations throughout
- Property-based testing for correctness
- Complete documentation and specifications"

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    echo ""
    echo "✅ GitHub CLI detected"
    echo "🚀 Creating repository and pushing..."
    
    if [ "$visibility" = "private" ]; then
        gh repo create "$repo_name" --private --source=. --remote=origin --push
    else
        gh repo create "$repo_name" --public --source=. --remote=origin --push
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Success! Repository created and code pushed!"
        echo "🌐 View at: https://github.com/$github_username/$repo_name"
    else
        echo ""
        echo "❌ Failed to create repository with GitHub CLI"
        echo "Please create the repository manually on GitHub"
    fi
else
    echo ""
    echo "⚠️  GitHub CLI not found"
    echo ""
    echo "📝 Manual steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository named: $repo_name"
    echo "3. Choose visibility: $visibility"
    echo "4. DO NOT initialize with README, .gitignore, or license"
    echo "5. Click 'Create repository'"
    echo ""
    echo "Then run these commands:"
    echo ""
    echo "  git remote add origin https://github.com/$github_username/$repo_name.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
    echo ""
fi

echo ""
echo "📚 Next steps:"
echo "1. Update README.md with your information"
echo "2. Add topics to your repository (python, manim, pdf, animation, llm)"
echo "3. Set up GitHub Actions for automated testing (optional)"
echo "4. Configure API keys for production use (see SETUP.md)"
echo ""
echo "✨ Done!"
