# Git Configuration Commands

## Check Current Configuration

```bash
# Check your name
git config --global user.name

# Check your email
git config --global user.email

# See all global config
git config --global --list

# See all config (global + local)
git config --list
```

## Set Up Your Identity (Required Before First Commit)

```bash
# Set your name (shows in commits)
git config --global user.name "Your Name"

# Set your email (should match GitHub email)
git config --global user.email "your.email@example.com"
```

**Example:**
```bash
git config --global user.name "John Doe"
git config --global user.email "john.doe@example.com"
```

## Other Useful Config Settings

```bash
# Set default branch name to 'main' instead of 'master'
git config --global init.defaultBranch main

# Enable colored output (easier to read)
git config --global color.ui auto

# Set default editor (for commit messages)
git config --global core.editor "nano"  # or vim, code, etc.

# Cache credentials for 1 hour (avoid repeated login)
git config --global credential.helper 'cache --timeout=3600'
```

## Verify Your Configuration

```bash
# Check what you set
git config --global --list

# Should show:
# user.name=Your Name
# user.email=your.email@example.com
# init.defaultbranch=main
# color.ui=auto
```

## Local vs Global Config

**Global** (`--global`): Applies to all git repos on your system
```bash
git config --global user.name "Your Name"
```

**Local** (`--local`): Applies only to current repository
```bash
git config --local user.name "Different Name"
```

**System** (`--system`): Applies system-wide (requires sudo)
```bash
sudo git config --system user.name "System Name"
```

Priority: Local > Global > System

## Unset a Configuration

```bash
# Remove global setting
git config --global --unset user.name

# Remove local setting
git config --local --unset user.name
```

## Edit Config File Directly

```bash
# Edit global config file
nano ~/.gitconfig

# Edit local config file (in repo)
nano .git/config
```

## Quick Setup for This Project

```bash
# 1. Set your identity (REQUIRED - do this first!)
git config --global user.name "Your Actual Name"
git config --global user.email "your.github.email@example.com"

# 2. Set default branch name
git config --global init.defaultBranch main

# 3. Enable colors
git config --global color.ui auto

# 4. Cache credentials for 1 hour
git config --global credential.helper 'cache --timeout=3600'

# 5. Verify everything
git config --global --list
```

---

**Important**: Use the same email as your GitHub account so your commits are properly linked to your profile!
