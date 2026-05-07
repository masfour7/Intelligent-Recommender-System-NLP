#!/bin/bash
# Run this once before pushing to GitHub.
# It removes Replit-internal files from git tracking (does NOT delete them locally).

echo "Removing Replit internals from git tracking..."
git rm --cached .replit replit.md uv.lock 2>/dev/null
git rm --cached -r .local/ .agents/ .cache/ 2>/dev/null
git rm --cached ".ipynb_checkpoints/demo-checkpoint.ipynb" 2>/dev/null
git rm --cached main.py pyproject.toml 2>/dev/null

echo "Done. Now commit and push:"
echo "  git add -A"
echo "  git commit -m 'Initial commit'"
echo "  git push"
