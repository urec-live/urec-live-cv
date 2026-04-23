# Railway Deployment Guide

This guide covers deploying urec-live-cv to [Railway](https://railway.app).

## Prerequisites

1. **Railway Account**: Sign up at [railway.app](https://railway.app)
2. **Git Repository**: Ensure the project is pushed to a Git repository (GitHub, GitLab, etc.)
3. **Railway CLI** (optional): Install via `npm install -g @railway/cli`

## Quick Start (Web UI)

### 1. Connect Your Repository

1. Log in to [Railway Dashboard](https://railway.app/dashboard)
2. Click **New Project** → **Deploy from GitHub**
3. Select your repository and branch
4. Railway will auto-detect the `Dockerfile` and deploy

### 2. Configure Environment Variables

In the Railway project settings, add any required environment variables:

| Variable | Default | Notes |
|----------|---------|-------|
| `PORT` | `8000` | Railway will set this automatically |
| `ENVIRONMENT` | `production` | Set to `development` to enable auto-reload (not recommended for production) |

### 3. Monitor Deployment

- Watch the build and deployment logs in the Railway dashboard
- Once deployed, you'll get a public URL like `https://urec-live-cv-production.up.railway.app`

## Advanced: CLI Deployment

### 1. Install Railway CLI

```bash
npm install -g @railway/cli
# or
brew install railway
```

### 2. Link Project

```bash
cd urec-live-cv
railway login
railway link  # Create a new project or link existing
```

### 3. Deploy

```bash
railway up
```

### 4. View Logs

```bash
railway logs
```

## Integration with UREC Live Backend

Once deployed on Railway, update your UREC Live backend configuration:

### Option A: Environment Variable

Set in the backend service (or `.env`):

```properties
APP_CV_SECONDARY_ENABLED=true
APP_CV_SECONDARY_BASE_URL=https://urec-live-cv-production.up.railway.app
APP_CV_SECONDARY_CONFIDENCE_THRESHOLD=0.65
APP_CV_SECONDARY_CACHE_TTL_MS=3000
```

### Option B: Railway Networking (If Both on Railway)

If the backend is also on Railway in the same environment:

```properties
APP_CV_SECONDARY_ENABLED=true
APP_CV_SECONDARY_BASE_URL=http://urec-live-cv:8000
APP_CV_SECONDARY_CONFIDENCE_THRESHOLD=0.65
APP_CV_SECONDARY_CACHE_TTL_MS=3000
```

Then, in the backend's Railway project settings, add a service reference to urec-live-cv.

## Troubleshooting

### Build Fails: "ModuleNotFoundError"

- Ensure all dependencies are in `requirements.txt`
- Run locally: `pip install -r requirements.txt && python scripts/run_api.py`

### Health Check Fails

- The health endpoint (`GET /health`) may not be responding
- Check Railway logs: `railway logs`
- Verify the API is listening on port 8000 (or the PORT env var set by Railway)

### Slow Model Downloads

- YOLO models (`yolov8x.pt`, `yolov8x-pose.pt`) are auto-downloaded on first run
- Initial deployment may take 5-10 minutes for the first request
- Consider pre-building a custom Docker image with models included

### Memory Issues

- The YOLO models require significant memory (~2GB)
- Railway's default plan includes 8GB memory per dyno, which is sufficient
- If needed, upgrade to a larger instance in Railway settings

## Custom Docker Image

If Railway's auto-detection isn't working as expected, ensure these files exist:

- **Dockerfile** ✓ (included)
- **.dockerignore** ✓ (included)
- **railway.json** ✓ (included)
- **Procfile** ✓ (included)

Railway will use the Dockerfile first. If it doesn't find one, it will try `Procfile` or a buildpack.

## Monitoring & Health

### Health Endpoint

The API exposes a health check:

```bash
curl https://<your-railway-url>/health
# Response: {"ok": true}
```

### Equipment Status Endpoints

- **GET** `/equipment/status` — Fetch all current equipment statuses
- **POST** `/equipment/status` — Update a single equipment status

Example:

```bash
curl https://<your-railway-url>/equipment/status
```

## Cost Considerations

- Railway provides **$5/month** free credits
- Simple FastAPI apps typically cost $0-2/month
- YOLO inference is CPU-intensive; monitor usage and upgrade if needed

## Next Steps

1. Deploy on Railway (steps above)
2. Test the `/health` endpoint
3. Update backend `APP_CV_SECONDARY_BASE_URL` to your Railway URL
4. Monitor equipment status predictions in the main UREC Live app

## Support

For Railway-specific issues:
- [Railway Docs](https://docs.railway.app)
- [Railway Community Discord](https://discord.gg/railway)

For urec-live-cv issues:
- Check logs: `railway logs`
- Review `src/backend/main.py` and `src/gymcv/` modules
