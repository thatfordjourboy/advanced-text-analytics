# 🚀 Emotion Detection Project - Deployment Guide

## **📋 Overview**
This guide covers deploying your emotion detection project to:
- **Frontend**: Vercel (React app)
- **Backend**: Render (FastAPI + ML pipeline)

## **⚠️ Important: Data Files**
Your project requires large data files that are **NOT in git**:
- **GloVe vectors**: `glove.2024.wikigiga.100d.zip` (~1.8GB)
- **GloVe text file**: `wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt` (~3GB)

## **🔧 Backend Deployment (Render)**

### **Step 1: Prepare Data Files**
```bash
# Download GloVe vectors to your local machine
cd backend/data/

# Option A: Download from Stanford (if available)
wget https://nlp.stanford.edu/data/glove.2024.wikigiga.100d.zip

# Option B: Use your existing local files
# Copy from your local development environment
```

### **Step 2: Render Deployment**
1. **Connect your GitHub repo** to Render
2. **Create a Web Service** with these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.12

### **Step 3: Handle Data Files in Production**
**Option A: Upload During Build (Recommended)**
```bash
# Add to your Render build script
# This downloads data during deployment
curl -L -o data/glove.2024.wikigiga.100d.zip \
  "https://your-storage-url/glove.2024.wikigiga.100d.zip"
```

**Option B: Use Render's File System**
- Upload data files via Render's dashboard
- Place in `/app/data/` directory

**Option C: External Storage (Most Professional)**
- Store on AWS S3, Google Cloud, or Hugging Face
- Download during deployment

## **🌐 Frontend Deployment (Vercel)**

### **Step 1: Connect Repository**
1. **Import your GitHub repo** to Vercel
2. **Set build settings**:
   - **Framework Preset**: Next.js (or React)
   - **Build Command**: `npm run build`
   - **Output Directory**: `frontend/build`

### **Step 2: Environment Variables**
Set these in Vercel:
```env
REACT_APP_API_URL=https://your-backend-url.onrender.com
REACT_APP_ENVIRONMENT=production
```

## **📁 File Structure for Deployment**

### **Backend (Render)**
```
/app/
├── app/                    # Your FastAPI code
├── data/                   # Data directory
│   ├── dialogues.json      # ✅ In git
│   ├── ontology.json       # ✅ In git
│   ├── glove.2024.wikigiga.100d.zip  # ❌ Need to add
│   └── wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt  # ❌ Need to add
├── models/                 # ✅ In git
├── requirements.txt        # ✅ In git
└── Dockerfile             # ✅ In git
```

### **Frontend (Vercel)**
```
/frontend/
├── src/                    # ✅ In git
├── public/                 # ✅ In git
├── package.json            # ✅ In git
└── build/                  # Generated during build
```

## **🔑 Environment Variables**

### **Backend (Render)**
```env
ENVIRONMENT=production
DEBUG=false
PORT=$PORT
```

### **Frontend (Vercel)**
```env
REACT_APP_API_URL=https://your-backend-url.onrender.com
REACT_APP_ENVIRONMENT=production
```

## **📊 Data File Management Strategies**

### **Strategy 1: Download During Build (Recommended)**
```bash
# Add to your deployment script
#!/bin/bash
cd /app/data

# Download GloVe vectors if not present
if [ ! -f "glove.2024.wikigiga.100d.zip" ]; then
    echo "Downloading GloVe vectors..."
    curl -L -o glove.2024.wikigiga.100d.zip \
      "https://your-storage-url/glove.2024.wikigiga.100d.zip"
fi

# Extract if needed
if [ ! -f "wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_050_combined.txt" ]; then
    echo "Extracting GloVe vectors..."
    unzip -o glove.2024.wikigiga.100d.zip
fi
```

### **Strategy 2: External Storage**
```python
# In your embeddings.py
import boto3
import os

def download_glove_vectors():
    """Download GloVe vectors from S3 if not present"""
    if not os.path.exists('data/glove.2024.wikigiga.100d.zip'):
        s3 = boto3.client('s3')
        s3.download_file('your-bucket', 'glove.2024.wikigiga.100d.zip', 
                        'data/glove.2024.wikigiga.100d.zip')
```

## **🚨 Common Issues & Solutions**

### **Issue: Build Timeout on Render**
**Solution**: Use external storage, don't download during build

### **Issue: Memory Limit Exceeded**
**Solution**: Process data in chunks, use streaming

### **Issue: Frontend Can't Connect to Backend**
**Solution**: Check CORS settings and environment variables

## **✅ Deployment Checklist**

### **Backend (Render)**
- [ ] Repository connected
- [ ] Data files available (downloaded or uploaded)
- [ ] Environment variables set
- [ ] Build successful
- [ ] Health check endpoint working

### **Frontend (Vercel)**
- [ ] Repository connected
- [ ] Environment variables set
- [ ] Build successful
- [ ] Can connect to backend API

## **🔗 Useful Links**
- [Render Documentation](https://render.com/docs)
- [Vercel Documentation](https://vercel.com/docs)
- [Stanford GloVe 2024 Vectors](https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.100d.zip)

## **📞 Support**
If you encounter issues:
1. Check Render/Vercel logs
2. Verify data files are accessible
3. Test API endpoints locally first
4. Check environment variables

---

**🎯 Your project is correctly structured!** The large data files stay out of git but are handled during deployment.
