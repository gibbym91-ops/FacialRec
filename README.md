# FaceMatch AI — Single File Browser App

A complete facial recognition comparison tool in a **single HTML file**. No backend, no API keys, no server needed. Everything runs in the browser.

## 🚀 Usage

### Option 1: Just open it
Double-click `index.html` — but note that loading models from CDN requires an internet connection, and some browsers block cross-origin requests when running from `file://`.

### Option 2: Serve locally (recommended)
```bash
# Python 3
python -m http.server 8080
# Then open http://localhost:8080

# Node.js
npx serve .
# Then open http://localhost:3000
```

### Option 3: Deploy to Vercel / Netlify / GitHub Pages
Just upload the `index.html` file. No build step, no configuration needed.

**Vercel:**
```bash
# Create vercel.json alongside index.html
echo '{"rewrites":[{"source":"/(.*)", "destination":"/index.html"}]}' > vercel.json
vercel
```

**Netlify:** Drag the folder into netlify.com/drop

**GitHub Pages:** Push to a repo, enable Pages from Settings → Pages

---

## 🧠 How It Works

All processing is done in the browser using face-api.js (built on TensorFlow.js):

1. **SSD MobileNetV1** — Face detection with confidence scoring
2. **68-Point Landmark CNN** — Maps jaw, eyes, brows, nose, mouth keypoints
3. **ResNet-34 Feature Embedding** — 128-dimensional face descriptor (99.38% LFW accuracy)
4. **Euclidean L2 Distance** — Match threshold calibrated at 0.600
5. **Per-Region Geometric Analysis** — Compares 6 facial zones independently
6. **Algorithmic Forensic Synopsis** — 4-paragraph analysis generated from the scores

**Models load from CDN (~12MB total)** — cached after first load.

## 🔒 Privacy

- **No data ever leaves your device**
- Images are processed entirely in-browser memory
- No server calls, no analytics, no tracking

## ⚠️ Notes on URL Mode

When loading images via URL, the remote server must allow cross-origin requests (`Access-Control-Allow-Origin: *`). Many sites block this. For best results, use the upload mode.
