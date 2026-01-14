# Setting Up /teleport for Cross-Device Access

## Overview

`/teleport` allows you to sync your project and Claude Code chat session across devices. This requires:
1. Your project code on a remote Git repository (GitHub/GitLab/etc)
2. The teleport feature enabled in Claude Code

## Step-by-Step Setup

### Step 1: Create a GitHub Repository

1. **Go to GitHub**: https://github.com/new
2. **Create repository**:
   - Name: `GallingModel` (or your preferred name)
   - Description: "PINN-based galling prediction model"
   - Visibility: Choose **Private** (recommended) or Public
   - **Don't** initialize with README, .gitignore, or license (we have these already)
3. **Click**: "Create repository"

### Step 2: Link Your Local Project to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Set your git username and email (one-time setup)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Add all files to git
git add .

# Create first commit
git commit -m "Initial commit: PINN galling prediction models

- Three model architectures: Feedforward, CNN-Hybrid, Physics-Only
- Complete training pipeline with two-stage optimization
- Automated plotting and model comparison
- Comprehensive documentation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Rename branch to main (GitHub standard)
git branch -M main

# Link to your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/GallingModel.git

# Push to GitHub
git push -u origin main
```

**Replace** `YOUR_USERNAME` with your actual GitHub username!

### Step 3: Verify Upload

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/GallingModel`
2. You should see all your files uploaded
3. Check that README.md displays correctly

### Step 4: Use /teleport

Now you can use `/teleport` in Claude Code:

**On Current Device:**
```bash
# In your Claude Code session, run:
/teleport
```

This will:
- Generate a unique teleport code
- Sync your current chat session
- Allow you to continue on another device

**On New Device:**
1. Install Claude Code on the new device
2. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GallingModel.git
   cd GallingModel
   ```
3. Start Claude Code and use the teleport code:
   ```bash
   claude-code
   /teleport YOUR_TELEPORT_CODE
   ```

## Alternative: GitLab or Other Git Hosts

If you prefer GitLab, Bitbucket, or self-hosted git:

**GitLab:**
```bash
git remote add origin https://gitlab.com/YOUR_USERNAME/GallingModel.git
git push -u origin main
```

**Bitbucket:**
```bash
git remote add origin https://bitbucket.org/YOUR_USERNAME/gallingmodel.git
git push -u origin main
```

## Important Notes

### What Gets Synced
✅ **Code files** (all .py, .yaml, .md files)
✅ **Configuration** (config/ directory)
✅ **Documentation** (all .md files)
✅ **Scripts** (scripts/, experiments/)

❌ **NOT synced** (in .gitignore):
- Large model checkpoints (*.pth files)
- Results directories (results/)
- Python cache (__pycache__)
- Environment files (.env)

### Syncing Results/Checkpoints

If you want to sync trained models and results:

**Option 1: Git LFS (Large File Storage)**
```bash
# Install git-lfs
sudo apt-get install git-lfs
git lfs install

# Track large files
git lfs track "*.pth"
git lfs track "results/**/*.png"

git add .gitattributes
git commit -m "Add Git LFS for model checkpoints"
git push
```

**Option 2: Cloud Storage**
- Use Google Drive, Dropbox, or AWS S3 for large files
- Keep code in git, data/results in cloud storage

### Security Considerations

**For Private Research:**
- Use **Private** repository on GitHub
- Don't commit sensitive data or credentials
- Add `.env` files to `.gitignore`

**For Public Sharing:**
- Review all files before making public
- Ensure no proprietary data in code
- Consider a license (MIT, Apache, etc.)

## Troubleshooting

### Authentication Issues

If git push asks for credentials:

**HTTPS (recommended):**
```bash
# GitHub now requires Personal Access Token (PAT)
# 1. Go to: https://github.com/settings/tokens
# 2. Generate new token (classic)
# 3. Use token as password when pushing
```

**SSH (alternative):**
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub: https://github.com/settings/keys
# Change remote URL
git remote set-url origin git@github.com:YOUR_USERNAME/GallingModel.git
```

### Large Files Rejected

If GitHub rejects files over 100MB:
```bash
# Remove large files from git
git rm --cached results/*.pth
echo "*.pth" >> .gitignore
git add .gitignore
git commit -m "Exclude large model files"
```

## Quick Reference

```bash
# Daily workflow
git add .
git commit -m "Description of changes"
git push

# On other device
git pull

# Check status
git status

# View history
git log --oneline

# Use teleport
/teleport                    # Generate code
/teleport YOUR_CODE_HERE     # Resume session
```

---

**Next Steps After Setup:**
1. Create GitHub repository
2. Run the git commands above
3. Use `/teleport` to get sync code
4. Access on other device with the code

Happy coding across devices! 🚀
