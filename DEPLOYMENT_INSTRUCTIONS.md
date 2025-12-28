# 🌐 PharmaPredictAI - Deployment Guide

## 🚀 Deploy to Render (Free & Easy)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### Step 2: Deploy on Render
1. Go to https://render.com (Sign up free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository: `Gihan007/Reaserch_papaer_Implement_for_Predicting_Drug-sales-data-for-specific-areas-in-the-srilanka-IEEE-`
4. Configure:
   - **Name**: `pharma-predict-ai`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **"Create Web Service"**

⏱️ **Wait 5-10 minutes** for deployment

🎉 **Your URL**: `https://pharma-predict-ai.onrender.com`

---

## 🔥 Alternative: Ngrok (Instant, No Code Changes)

Perfect for **quick testing** without deployment:

### Step 1: Download Ngrok
```bash
# Download from: https://ngrok.com/download
# Or install: choco install ngrok (Windows)
```

### Step 2: Run Your App
```bash
python app.py
```

### Step 3: Expose to Internet
```bash
ngrok http 5000
```

✅ **Instant Public URL**: `https://abc123.ngrok.io`

**Send this URL to your friend** - works immediately!

⚠️ URL changes each time you restart ngrok (Free tier)

---

## ☁️ Alternative: Railway (Free $5 Credit)

1. Go to https://railway.app
2. Click **"Start a New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your repository
5. Railway auto-detects Flask and deploys

**Your URL**: `https://pharma-predict-ai.up.railway.app`

---

## 🐳 Alternative: Heroku (Paid but Reliable)

```bash
heroku login
heroku create pharma-predict-ai
git push heroku main
```

---

## 📊 Best Option for You

| Method | Speed | Free Forever | Best For |
|--------|-------|-------------|----------|
| **Ngrok** ⭐ | Instant | Yes (with restarts) | Quick testing, demos |
| **Render** | 10 mins | Yes | Permanent public URL |
| **Railway** | 5 mins | $5 credit | Fast deployment |

## 🎯 Recommended: Use Ngrok Now, Deploy to Render Later

**For immediate testing:**
```bash
ngrok http 5000
```
Send the ngrok URL to your friend right now!

**For permanent deployment:**
Push to GitHub and deploy on Render (free forever)

---

## 📝 Files Created for Deployment
- ✅ `Procfile` - Tells hosting how to run your app
- ✅ `render.yaml` - Render configuration
- ✅ Updated `requirements.txt` - Fixed versions
- ✅ Updated `app.py` - Production-ready settings
