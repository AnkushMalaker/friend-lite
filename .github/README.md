# GitHub Actions CI/CD Setup for Friend Lite

This sets up **automatic GitHub releases** with APK/IPA files whenever you push code.

## 🚀 How This Works

1. You push code to GitHub
2. GitHub automatically builds **both Android APK and iOS IPA**
3. **Creates GitHub Releases** with both files attached
4. You download directly from the **Releases** tab!

## 🎯 Quick Setup (2 Steps)

### Step 1: Get Expo Token
1. Go to [expo.dev](https://expo.dev) and sign in/create account
2. Go to [Access Tokens](https://expo.dev/accounts/[account]/settings/access-tokens)
3. Create a new token and copy it

### Step 2: Add GitHub Secret
1. In your GitHub repo: **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `EXPO_TOKEN`
4. Value: Paste your token from Step 1
5. Click **Add secret**

## ⚡ That's It!