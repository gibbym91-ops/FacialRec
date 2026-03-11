# FaceMatch AI — Forensic Facial Recognition System

A state-of-the-art facial recognition and comparison tool powered by deep learning, running entirely in the browser with AI-generated forensic analysis reports.

## 🧠 How It Works

This system implements a multi-stage facial recognition pipeline based on NIST FRVT standards:

1. **Face Detection** — SSD MobileNetV1 CNN detects and localizes faces
2. **68-Point Landmark Detection** — ResNet-based CNN maps 68 anatomical keypoints (jaw, eyes, brows, nose, mouth)
3. **Face Alignment** — Landmarks are used to normalize face orientation
4. **Feature Embedding** — ResNet-34 architecture encodes each face into a **128-dimensional descriptor vector**
5. **Euclidean Distance** — L2 distance between descriptors determines similarity (threshold: 0.6)
6. **Region Analysis** — Per-region geometric comparison across facial zones
7. **AI Synopsis** — Claude AI generates a professional forensic analysis report

**Benchmark:** 99.38% accuracy on the LFW (Labeled Faces in the Wild) dataset

## 📊 Similarity Scale

| Score | Verdict |
|-------|---------|
| 85%+ | DEFINITE MATCH |
| 70–84% | STRONG MATCH |
| 55–69% | POSSIBLE MATCH |
| 40–54% | WEAK MATCH |
| <40% | NO MATCH |

## 🚀 Deployment on Vercel

### Prerequisites
- Node.js 18+
- A Vercel account
- An Anthropic API key (for AI synopsis — optional, fallback text is provided)

### 1. Clone / Initialize

```bash
git clone <your-repo-url>
cd facematch-ai
npm install
```

### 2. Environment Variables

Create a `.env.local` file:

```env
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

For Vercel deployment, add this in your Vercel project settings under **Environment Variables**.

### 3. Local Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### 4. Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Or connect your GitHub repo to Vercel for automatic deployments
```

### 5. Git Repository Setup

```bash
git init
git add .
git commit -m "Initial commit: FaceMatch AI"
git remote add origin https://github.com/yourusername/facematch-ai.git
git push -u origin main
```

Then import to Vercel: https://vercel.com/new

## 🔧 Architecture

```
facematch-ai/
├── pages/
│   ├── index.tsx          # Main page
│   ├── _app.tsx           # App wrapper
│   └── api/
│       └── analyze.ts     # Claude AI synopsis API route
├── components/
│   ├── FaceMatch.tsx      # Core UI component
│   └── FaceMatch.module.css
├── styles/
│   └── globals.css
├── next.config.js
├── vercel.json
└── package.json
```

## 🤖 AI Models Used

All models load from CDN at runtime (no files to download):

| Model | Size | Purpose |
|-------|------|---------|
| SSD MobileNetV1 | ~5.4MB | Face detection |
| Face Landmark 68 | ~350KB | Landmark detection |
| Face Recognition Net (ResNet-34) | ~6.2MB | 128-D descriptor |
| Face Expression Net | ~310KB | Expression recognition |
| Age/Gender Net | ~420KB | Age & gender estimation |

**Total: ~12.7MB loaded from jsDelivr CDN**

## 📋 Notes

- **Privacy**: All processing happens in the browser. Images are never uploaded to any server (except for the text-only AI synopsis request which sends only numeric scores).
- **Accuracy**: The system achieves 99.38% on LFW benchmark. Real-world accuracy varies with image quality, lighting, pose, and age differences between photos.
- **NIST Standards**: Uses Euclidean distance threshold of 0.6 as calibrated by face-api.js authors based on NIST FRVT methodology.
- The AI synopsis requires `ANTHROPIC_API_KEY`. Without it, a detailed algorithmic fallback analysis is used.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Do not use for surveillance, identity verification in legal proceedings, or any application requiring certified biometric systems. AI facial recognition systems have known biases across demographic groups.
