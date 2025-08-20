# 🚀 Render Deployment Guide for Emotion Detection Backend

## 📋 Prerequisites

- Render account (paid subscription recommended)
- GitHub repository connected to Render
- Backend code pushed to GitHub

## 🎯 Deployment Strategy

Since your app requires **2.2GB of data files**, we use **startup-time downloads** with **persistent disk storage**.

## 📁 Required Data Files

| File | Size | Purpose | Source |
|------|------|---------|---------|
| `glove.2024.wikigiga.100d.zip` | 555MB | GloVe embeddings | Stanford NLP |
| `wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt` | 1.6GB | Extracted vectors | Extracted from zip |
| `dialogues.json` | 45MB | Emotion dataset | Your dataset |
| `ontology.json` | 1.4KB | Emotion labels | Your labels |

## 🚀 Step-by-Step Deployment

### 1. **Connect Repository to Render**

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select the `emotion-detection` repository

### 2. **Configure Service Settings**

```yaml
Name: emotion-detection-backend
Environment: Python
Region: Choose closest to you
Branch: main
Root Directory: backend
```

### 3. **Build & Start Commands**

```bash
# Build Command (Simple):
pip install --upgrade pip
pip install -r requirements.txt

# Start Command:
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 4. **Environment Variables**

```bash
PYTHON_VERSION=3.11
ENVIRONMENT=production
DEBUG=false
PYTHONPATH=/app
```

### 5. **Advanced Settings**

- **Instance Type**: Standard-1x (recommended for data processing)
- **Health Check Path**: `/health`

## 🔧 How It Works

### **First Deployment:**
1. **Fast build** (just Python dependencies)
2. **Service starts** and runs startup script
3. **Downloads** GloVe vectors (555MB) during startup
4. **Extracts** vectors to 1.6GB text file
5. **Creates** sample data files if needed
6. **Loads** embeddings and dataset
7. **Service ready** for requests

### **Subsequent Deployments:**
1. **Fast build** (dependencies only)
2. **Service starts** and checks files
3. **Files exist** on persistent disk - skip download
4. **Service ready** immediately

## ⚠️ Important Notes

### **First Startup Time:**
- **Data download**: 10-15 minutes (555MB + extraction)
- **Total startup**: 15-20 minutes
- **Subsequent starts**: 30-60 seconds

### **Storage:**
- **Persistent disk**: Files survive restarts and redeploys
- **Download once**: Files cached permanently
- **No rebuild needed**: Data files persist across deployments

### **Cost:**
- **Standard-1x**: $7/month (recommended)
- **Persistent disk**: Included in plan

## 🚨 Troubleshooting

### **Startup Takes Too Long:**
- **First time**: Normal (downloading 2.2GB)
- **Subsequent**: Check logs for file access issues

### **Startup Fails - Download Error:**
- Check startup logs in Render dashboard
- Stanford servers might be down
- Try manual file upload via Render dashboard

### **Files Not Found:**
- Check startup script logs
- Verify persistent disk is working
- Check file permissions

### **Health Check Fails:**
- Check application logs
- Verify startup script completed
- Check PORT environment variable

## 🔍 Monitoring

### **Startup Logs:**
- Watch download progress
- Verify file extraction
- Check for errors

### **Runtime Logs:**
- Monitor application startup
- Check data loading success
- Verify health check responses

## 📊 Expected Results

### **Successful Deployment:**
- ✅ Service shows "Live" status
- ✅ Health check passes
- ✅ API endpoints respond
- ✅ Models load successfully

### **Performance:**
- **First startup**: 15-20 minutes (downloading data)
- **Subsequent starts**: 30-60 seconds
- **API response**: <100ms (after warm-up)

## 🎉 Success!

Once deployed, your backend will be available at:
```
https://your-service-name.onrender.com
```

Test with:
```bash
curl https://your-service-name.onrender.com/health
```

## 🔄 Updates

- **Code changes**: Auto-deploy on Git push
- **Data file updates**: Automatic (startup script handles)
- **Dependency updates**: Auto-update on requirements.txt change

## 🆚 Advantages of This Approach

### **Over Build-Time Downloads:**
- ✅ **Faster builds** (no 2.2GB download each build)
- ✅ **Persistent storage** (files survive redeploys)
- ✅ **Smart caching** (download once, use forever)
- ✅ **Multiple fallbacks** (if one source fails)
- ✅ **Progress tracking** (real-time download status)

---

**Need help?** Check the startup logs and runtime logs in Render dashboard for detailed information.
