# RENDER DEPLOYMENT GUIDE

## Problem: Model Not Loading on Railway

The issue is that your model files (33MB+) are too large for Railway's free tier or aren't being deployed properly.

## Solution: Deploy on Render

Render handles large files better and supports custom build scripts.

---

## STEP 1: UPLOAD MODEL FILES TO GOOGLE DRIVE

### 1.1 Upload Your Models

1. Go to [Google Drive](https://drive.google.com)
2. Create a folder called `crop-pest-models`
3. Upload these files:
   - `crop_pest_model_finetuned.h5` (from your local `models/` folder)
   - `class_names.json` (from your local `models/` folder)

### 1.2 Get Shareable Links

For each file:

1. Right-click → **Get link** → **Anyone with the link can view**
2. Copy the link (looks like: `https://drive.google.com/file/d/1ABC123XYZ/view?usp=sharing`)
3. Extract the FILE_ID (the part between `/d/` and `/view`)
   - Example: `1ABC123XYZ` is the FILE_ID

---

## STEP 2: COMMIT AND PUSH RENDER CONFIGURATION

```bash
# Make build.sh executable
git update-index --chmod=+x build.sh

# Add all files
git add build.sh render.yaml download_models.py requirements.txt

# Commit
git commit -m "Add Render deployment configuration with model download"

# Push
git push origin main
```

---

## STEP 3: DEPLOY ON RENDER

### 3.1 Sign Up / Login

1. Go to [render.com](https://render.com)
2. Click **"Get Started"** → **Sign up with GitHub**

### 3.2 Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub account
3. Select repository: **`crop-pest-detection`**
4. Click **"Connect"**

### 3.3 Configure Service

Fill in the following:

**Name:** `crop-pest-detection-api`

**Region:** Choose closest to your users

**Branch:** `main`

**Runtime:** `Python 3`

**Build Command:** `./build.sh`

**Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

**Instance Type:** `Free` (or paid for better performance)

### 3.4 Add Environment Variables

Click **"Advanced"** → **"Add Environment Variable"**

Add these variables:

```
PYTHON_VERSION=3.11
MODEL_PATH=models/crop_pest_model_finetuned.h5
CLASS_NAMES_PATH=models/class_names.json
MODEL_FILE_ID=1XYduXt8lOKPoCKT6nhYZUuo_GrCwCTWy
CLASS_NAMES_FILE_ID=1u2xMlx3wOer2-e_Qo7oyru_Nwk9n4R7S
```

**IMPORTANT:** Replace with your actual Google Drive FILE_IDs from Step 1.2

### 3.5 Deploy

1. Click **"Create Web Service"**
2. Wait 5-10 minutes for build and deployment
3. Watch the logs for any errors

---

## STEP 4: VERIFY DEPLOYMENT

### 4.1 Check Health Endpoint

Once deployed, visit:

```
https://your-app-name.onrender.com/health
```

You should see:

```json
{
  "status": "healthy",
  "uptime_seconds": 123.4,
  "model_loaded": true,  ← Should be TRUE now!
  "total_requests": 1,
  "average_latency_ms": 0.05
}
```

### 4.2 Test API Documentation

Visit:

```
https://your-app-name.onrender.com/docs
```

---

## STEP 5: UPDATE NETLIFY UI

Update your UI's environment variable on Netlify:

1. Go to Netlify dashboard
2. **Site settings** → **Environment variables**
3. Update `VITE_API_URL` to:
   ```
   https://your-app-name.onrender.com
   ```
4. **Trigger redeploy** of your UI

---

## ALTERNATIVE: USE SMALLER MODEL OR HUGGING FACE

If you don't want to use Google Drive:

### Option A: Use GitHub Release

1. Create a release on GitHub
2. Attach model files to release
3. Download from release URL in `download_models.py`

### Option B: Use Hugging Face Hub

1. Upload models to Hugging Face: https://huggingface.co
2. Use `huggingface_hub` library to download

### Option C: Reduce Model Size

```python
# In your training script, save with compression
model.save('model.h5', save_format='h5', compression='gzip')
```

---

## RENDER VS RAILWAY

| Feature            | Render              | Railway          |
| ------------------ | ------------------- | ---------------- |
| Free Tier          | 750 hours/month     | $5 credit/month  |
| Large Files        | Better support      | Limited          |
| Build Time         | 5-10 min            | 2-5 min          |
| Cold Starts        | ~30s (free tier)    | Faster           |
| Custom Build       | Excellent           | Good             |
| **Recommendation** | ✅ For large models | For smaller apps |

---

## TROUBLESHOOTING

### Build Fails

- Check logs in Render dashboard
- Ensure `build.sh` is executable
- Verify Google Drive links are public

### Model Still Not Loading

- Check environment variables are set correctly
- Verify FILE_IDs are correct
- Check Render logs for download errors

### Out of Memory

- Upgrade to paid tier ($7/month)
- Or use smaller model
- Or optimize TensorFlow installation

---

## YOUR DEPLOYMENT CHECKLIST

- [ ] Upload models to Google Drive
- [ ] Get FILE_IDs from Google Drive links
- [ ] Commit Render configuration files
- [ ] Push to GitHub
- [ ] Create Render web service
- [ ] Add environment variables (including FILE_IDs)
- [ ] Deploy and wait for build
- [ ] Check `/health` endpoint (model_loaded: true)
- [ ] Update Netlify with new Render URL
- [ ] Test full application

---

**Your model will now load successfully on Render! 🎉**
