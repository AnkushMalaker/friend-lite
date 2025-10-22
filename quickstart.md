# Friend-Lite Quick Start

## What You're Building (Complete Beginners Start Here!)

You're setting up your own personal AI that:
- **Runs on your home computer** - processes audio, stores memories, runs AI models
- **Connects to your phone** - where you use the Friend-Lite app and OMI device
- **Works everywhere** - your phone can access your home AI from anywhere

Think of it like having Siri/Alexa, but it's **your own AI** running on **your hardware** with **your data**.

## The Setup (What You'll Install)

### On Your Home Computer
- **Docker** - Runs all the AI services (like having multiple apps in containers)
- **Friend-Lite Backend** - The main AI brain (transcription, memory, processing)
- **Tailscale** - Creates secure tunnel so your phone can reach home

### On Your Phone  
- **Tailscale** - Connects securely to your home computer
- **Friend-Lite Mobile App** - Interface for your OMI device and conversations

### AI Services (You'll Get API Keys)
- **Deepgram** - Converts speech to text (free credits available)
- **OpenAI** - Powers memory extraction and insights (pay per use)
- **Optional: Hugging Face** - For speaker recognition (free)

## Step 1: Install Required Software

### On Your Home Computer

**Git** (Downloads code from the internet):
- **Windows/Mac**: [Download Git](https://git-scm.com/downloads)  
- **Linux**: `sudo apt install git` or `sudo yum install git`

**Docker** (Runs the AI services):
- **Windows/Mac**: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: [Install Docker](https://docs.docker.com/engine/install/)
- **After install**: Make sure Docker Desktop is running

**uv** (Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
*This downloads and runs Python programs for you*

**Tailscale** (Connects your phone to home computer):
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```
*Follow the login prompts - this gives your computer a special IP address*

### On Your Phone

**Tailscale App**:
- **iPhone**: [App Store - Tailscale](https://apps.apple.com/app/tailscale/id1470499037)
- **Android**: [Google Play - Tailscale](https://play.google.com/store/apps/details?id=com.tailscale.ipn)

*Log in with the same account you used on your computer*

## Step 2: Get Your AI API Keys

You need accounts with AI services (most have free tiers):

### Deepgram (Speech-to-Text) - Required
1. Go to [console.deepgram.com](https://console.deepgram.com)
2. Sign up for free account (get $200 free credits)
3. Go to "API Keys" → Create new key
4. **Copy the key** - you'll need it in setup

### OpenAI (AI Brain) - Required  
1. Go to [platform.openai.com](https://platform.openai.com)
2. Create account and add payment method (typically costs $1-5/month)
3. Go to "API Keys" → Create new key
4. **Copy the key** - you'll need it in setup

### Hugging Face (Speaker Recognition) - Optional
1. Go to [huggingface.co](https://huggingface.co/join)
2. Create free account
3. Go to Settings → Access Tokens → Create new token
4. **Copy the token** - for identifying different speakers

## Step 3: Download and Setup Friend-Lite

### On Your Home Computer

**Download the code:**
```bash
git clone https://github.com/AnkushMalaker/friend-lite.git
cd friend-lite
```

**Run the setup wizard:**
```bash
uv run --with-requirements setup-requirements.txt python init.py
```

### What the Setup Wizard Will Ask You

The wizard will ask questions - here's what to answer:

**"Admin email"**: Your email (for logging into web dashboard)
**"Admin password"**: Password for web dashboard (8+ characters)

**"Choose transcription provider"**: Choose 1 (Deepgram)
**"Deepgram API key"**: Paste the key you got from Deepgram

**"Choose LLM provider"**: Choose 1 (OpenAI) 
**"OpenAI API key"**: Paste the key you got from OpenAI
**"OpenAI model"**: Keep default (gpt-4o-mini)

**"Enable Speaker Recognition"**: Say Yes if you got Hugging Face token
**"Hugging Face token"**: Paste your token (if you got one)

**"Enable HTTPS"**: **Say Yes** (needed for phone connection)
**"Server IP for SSL certificate"**: 
- Run `tailscale ip` in another terminal
- Copy the IP that starts with `100.` (like `100.64.1.5`)  
- Paste that IP here

*The wizard creates all the configuration files automatically*

**Start the services:**
```bash
uv run --with-requirements setup-requirements.txt python services.py start --all --build
```

*This downloads and starts all the AI services - takes 5-10 minutes first time*

## Step 4: Test Your Backend (Important!)

Before connecting your phone, make sure everything works:

1. Visit: **https://[your-tailscale-ip]** (like `https://100.64.1.5`)
   
   *Your browser will warn about "unsafe certificate" - click "Advanced" → "Proceed anyway"*

2. You should see the Friend-Lite dashboard
3. Click "Live Recording" in the sidebar
4. Test your microphone - record a short clip
5. Check that it gets transcribed and appears in "Conversations"
6. **Only proceed to phone setup when this works perfectly!**

## Step 5: Install Friend-Lite on Your Phone

**No development setup needed - just download and install!**

### Android Users
1. Go to [GitHub Releases](https://github.com/AnkushMalaker/friend-lite/releases)
2. Find the latest release and download `friend-lite-android.apk`
3. Install APK on your phone:
   - Enable "Install from unknown sources" in Android settings
   - Tap the downloaded APK file to install

### iPhone Users  
1. Go to [GitHub Releases](https://github.com/AnkushMalaker/friend-lite/releases)
2. Find the latest release and download `friend-lite-ios.ipa` 
3. Install using sideloading tool:
   - **AltStore** (recommended): [altstore.io](https://altstore.io)
   - **Sideloadly**: [sideloadly.io](https://sideloadly.io)
   
   *Note: iOS requires sideloading since we're not on App Store yet*

### Configure the App
1. **First**: Make sure Tailscale is running on your phone
2. Open Friend-Lite app
3. Go to Settings → Backend Configuration
4. Enter Backend URL: `https://[your-tailscale-ip]` 
   
   *Use the same IP as your web dashboard - like `https://100.64.1.5`*
   
5. Tap "Test Connection" - should show **green checkmark**
6. If connection fails, double-check:
   - Tailscale is running on phone
   - Same IP as web dashboard  
   - Using `https://` (not `http://`)

## Step 6: Connect Your OMI Device

1. Turn on your OMI/Friend device (make sure it's charged)
2. Open Friend-Lite app on your phone
3. Go to "Devices" tab → "Add New Device"
4. Follow Bluetooth pairing instructions
5. Once connected, start a conversation!
6. **Check your web dashboard** - the conversation should appear there

## You're Done! 🎉

**What you now have:**
- ✅ Personal AI running on your home computer
- ✅ Phone app connected securely via Tailscale  
- ✅ OMI device streaming audio to your AI
- ✅ All conversations processed privately and stored locally
- ✅ Access from anywhere via your phone

**Next steps:**
- Explore the web dashboard features
- Try voice commands and see memories get extracted
- Invite others to test conversations (if you enabled speaker recognition)

## Troubleshooting

### Setup Issues
- **"Command not found"**: Make sure Docker Desktop is running
- **"Permission denied"**: Try `sudo` before commands (Linux/Mac)
- **"uv not found"**: Restart terminal after installing uv

### Connection Issues  
- **Phone can't reach backend**: Check Tailscale is running on both devices
- **Certificate warnings**: Click "Advanced" → "Proceed" in browser
- **Test connection fails**: Verify you're using `https://` and correct Tailscale IP

### API Issues
- **Transcription not working**: Check Deepgram API key is correct
- **No memories created**: Check OpenAI API key and account has credits
- **High costs**: Switch to `gpt-4o-mini` model for cheaper processing

## Need Help?

- **Full Documentation**: [CLAUDE.md](CLAUDE.md) - Complete technical reference
- **Architecture Details**: [Docs/features.md](Docs/features.md) - How everything works  
- **Advanced Setup**: [Docs/init-system.md](Docs/init-system.md) - Power user options