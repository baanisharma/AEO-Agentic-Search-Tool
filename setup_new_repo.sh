#!/bin/bash

# AIO Search Tool - New Repository Setup Script
# Run this script after creating your new repository on GitHub

echo "🚀 Setting up new repository for AIO Search Tool..."

# Check if repository URL is provided
if [ -z "$1" ]; then
    echo "❌ Please provide your new repository URL as an argument"
    echo "Usage: ./setup_new_repo.sh git@github.com:username/repo-name.git"
    echo ""
    echo "📝 Steps to get your repository URL:"
    echo "1. Go to https://github.com and create a new repository"
    echo "2. Copy the SSH URL (git@github.com:username/repo-name.git)"
    echo "3. Run: ./setup_new_repo.sh git@github.com:username/repo-name.git"
    exit 1
fi

NEW_REPO_URL=$1

echo "📋 Current repository status:"
git status

echo ""
echo "🔗 Current remotes:"
git remote -v

echo ""
echo "🆕 Adding new repository as 'new-origin'..."
git remote add new-origin "$NEW_REPO_URL"

echo ""
echo "📤 Pushing all branches to new repository..."
git push new-origin main

echo ""
echo "✅ Successfully pushed to new repository!"
echo "🔗 New repository URL: $NEW_REPO_URL"

echo ""
read -p "🤔 Do you want to make this the primary origin? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Updating origin to point to new repository..."
    git remote remove origin
    git remote rename new-origin origin
    echo "✅ Origin updated! Your project now points to the new repository."
else
    echo "ℹ️  Old origin kept. You now have two remotes:"
    echo "   - origin: Your old repository"
    echo "   - new-origin: Your new repository"
fi

echo ""
echo "🎉 Setup complete! Your project is now available in the new repository."
echo "🔗 View your new repository: $NEW_REPO_URL" 