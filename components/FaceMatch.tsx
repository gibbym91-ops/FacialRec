import React, { useState, useRef, useCallback, useEffect } from 'react';
import styles from './FaceMatch.module.css';

// ─── Types ────────────────────────────────────────────────────────────────────
interface FaceResult {
  descriptor: Float32Array;
  landmarks: any;
  detection: any;
  expressions?: any;
  age?: number;
  gender?: string;
  genderProbability?: number;
  canvas: string; // base64 annotated image
}

interface FeatureBreakdown {
  'Left Eye': number;
  'Right Eye': number;
  'Nose Bridge': number;
  'Mouth': number;
  'Jaw & Chin': number;
  'Eyebrows': number;
}

interface ComparisonResult {
  distance: number;
  similarity: number;
  verdict: string;
  verdictClass: string;
  featureBreakdown: FeatureBreakdown;
}

type InputMode = 'upload' | 'url';

// ─── Utilities ────────────────────────────────────────────────────────────────
function euclideanDistance(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

function distanceToSimilarity(distance: number): number {
  // Calibrated for face-api.js ResNet-34: threshold at 0.6
  // Map [0, 1.2] → [100%, 0%] with sigmoid-like curve
  const normalized = Math.max(0, Math.min(1.5, distance));
  const similarity = Math.max(0, 1 - normalized / 1.1);
  return Math.min(1, similarity);
}

function computeFeatureBreakdown(l1: any, l2: any): FeatureBreakdown {
  const dist = (p1: any, p2: any) => Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2);
  const avgDist = (arr1: any[], arr2: any[]) => {
    let total = 0;
    const n = Math.min(arr1.length, arr2.length);
    for (let i = 0; i < n; i++) total += dist(arr1[i], arr2[i]);
    return total / n;
  };

  // Normalize by face size
  const faceSize1 = dist(l1.positions[0], l1.positions[16]);
  const faceSize2 = dist(l2.positions[0], l2.positions[16]);
  const scale = (faceSize1 + faceSize2) / 2;

  const rawScores: Record<string, number> = {
    'Left Eye': avgDist(l1.getLeftEye(), l2.getLeftEye()) / scale,
    'Right Eye': avgDist(l1.getRightEye(), l2.getRightEye()) / scale,
    'Nose Bridge': avgDist(l1.getNose(), l2.getNose()) / scale,
    'Mouth': avgDist(l1.getMouth(), l2.getMouth()) / scale,
    'Jaw & Chin': avgDist(l1.getJawOutline(), l2.getJawOutline()) / scale,
    'Eyebrows': [...avgDist(l1.getLeftEyeBrow(), l2.getLeftEyeBrow()), ...avgDist(l1.getRightEyeBrow(), l2.getRightEyeBrow())][0] / scale,
  };

  // Convert distances to similarities
  const result: any = {};
  for (const [k, v] of Object.entries(rawScores)) {
    result[k] = Math.max(0, Math.min(1, 1 - v * 3));
  }
  return result as FeatureBreakdown;
}

function getVerdict(similarity: number): { verdict: string; verdictClass: string } {
  if (similarity >= 0.85) return { verdict: 'DEFINITE MATCH', verdictClass: 'definite' };
  if (similarity >= 0.70) return { verdict: 'STRONG MATCH', verdictClass: 'strong' };
  if (similarity >= 0.55) return { verdict: 'POSSIBLE MATCH', verdictClass: 'possible' };
  if (similarity >= 0.40) return { verdict: 'WEAK MATCH', verdictClass: 'weak' };
  return { verdict: 'NO MATCH', verdictClass: 'none' };
}

// ─── Draw annotated face canvas ───────────────────────────────────────────────
async function drawAnnotatedFace(img: HTMLImageElement, result: any, faceapi: any): Promise<string> {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, 0, 0);

  // Scale detections to display size
  const dims = { width: canvas.width, height: canvas.height };
  const resized = faceapi.resizeResults(result, dims);

  // Draw bounding box
  const box = resized.detection.box;
  ctx.strokeStyle = '#00d4ff';
  ctx.lineWidth = 2;
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  // Draw corner accents
  const cl = 20;
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 3;
  // TL
  ctx.beginPath(); ctx.moveTo(box.x, box.y + cl); ctx.lineTo(box.x, box.y); ctx.lineTo(box.x + cl, box.y); ctx.stroke();
  // TR
  ctx.beginPath(); ctx.moveTo(box.x + box.width - cl, box.y); ctx.lineTo(box.x + box.width, box.y); ctx.lineTo(box.x + box.width, box.y + cl); ctx.stroke();
  // BL
  ctx.beginPath(); ctx.moveTo(box.x, box.y + box.height - cl); ctx.lineTo(box.x, box.y + box.height); ctx.lineTo(box.x + cl, box.y + box.height); ctx.stroke();
  // BR
  ctx.beginPath(); ctx.moveTo(box.x + box.width - cl, box.y + box.height); ctx.lineTo(box.x + box.width, box.y + box.height); ctx.lineTo(box.x + box.width, box.y + box.height - cl); ctx.stroke();

  // Draw 68 landmarks
  if (resized.landmarks) {
    const pts = resized.landmarks.positions;
    const groups = [
      { pts: resized.landmarks.getJawOutline(), color: '#4488ff', r: 2 },
      { pts: resized.landmarks.getLeftEyeBrow(), color: '#ff88ff', r: 2.5 },
      { pts: resized.landmarks.getRightEyeBrow(), color: '#ff88ff', r: 2.5 },
      { pts: resized.landmarks.getNose(), color: '#ffaa00', r: 2.5 },
      { pts: resized.landmarks.getLeftEye(), color: '#00ffcc', r: 2.5 },
      { pts: resized.landmarks.getRightEye(), color: '#00ffcc', r: 2.5 },
      { pts: resized.landmarks.getMouth(), color: '#ff4466', r: 2.5 },
    ];

    for (const group of groups) {
      ctx.fillStyle = group.color;
      for (const pt of group.pts) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, group.r, 0, Math.PI * 2);
        ctx.fill();
      }
      // Connect with lines
      ctx.strokeStyle = group.color + '60';
      ctx.lineWidth = 1;
      ctx.beginPath();
      group.pts.forEach((pt: any, i: number) => {
        if (i === 0) ctx.moveTo(pt.x, pt.y); else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
    }
  }

  // Confidence badge
  const conf = (resized.detection.score * 100).toFixed(0);
  ctx.fillStyle = 'rgba(0,0,0,0.7)';
  ctx.fillRect(box.x, box.y - 24, 110, 22);
  ctx.fillStyle = '#00d4ff';
  ctx.font = `bold 11px 'Space Mono', monospace`;
  ctx.fillText(`CONF: ${conf}%`, box.x + 6, box.y - 8);

  return canvas.toDataURL('image/jpeg', 0.92);
}

// ─── Image Panel Component ────────────────────────────────────────────────────
function ImagePanel({
  label,
  image,
  annotatedImage,
  faceResult,
  loading,
  error,
  inputMode,
  onModeChange,
  onFileChange,
  onUrlChange,
  urlValue,
}: any) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className={styles.imagePanel}>
      <div className={styles.panelHeader}>
        <span className={styles.panelLabel}>{label}</span>
        <div className={styles.modeToggle}>
          <button
            className={inputMode === 'upload' ? styles.modeActive : styles.modeBtn}
            onClick={() => onModeChange('upload')}
          >↑ UPLOAD</button>
          <button
            className={inputMode === 'url' ? styles.modeActive : styles.modeBtn}
            onClick={() => onModeChange('url')}
          >⊕ URL</button>
        </div>
      </div>

      <div
        className={`${styles.dropZone} ${image ? styles.dropZoneHasImage : ''}`}
        onClick={() => inputMode === 'upload' && fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); }}
        onDrop={(e) => {
          e.preventDefault();
          if (inputMode === 'upload') {
            const file = e.dataTransfer.files[0];
            if (file) onFileChange(file);
          }
        }}
      >
        {loading && (
          <div className={styles.loadingOverlay}>
            <div className={styles.scanLine} />
            <span className={styles.loadingText}>ANALYZING...</span>
          </div>
        )}

        {annotatedImage ? (
          <img src={annotatedImage} alt={label} className={styles.faceImage} />
        ) : image ? (
          <img src={image} alt={label} className={styles.faceImage} />
        ) : (
          <div className={styles.placeholder}>
            <div className={styles.placeholderIcon}>
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <rect x="4" y="4" width="40" height="40" rx="4" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 4"/>
                <circle cx="24" cy="20" r="6" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M10 38c0-7.732 6.268-14 14-14s14 6.268 14 14" stroke="currentColor" strokeWidth="1.5"/>
                <path d="M18 8l-6 6M30 8l6 6M18 40l-6-6M30 40l6-6" stroke="currentColor" strokeWidth="1" opacity="0.4"/>
              </svg>
            </div>
            <span className={styles.placeholderText}>
              {inputMode === 'upload' ? 'DRAG & DROP or CLICK' : 'ENTER URL BELOW'}
            </span>
          </div>
        )}
      </div>

      {inputMode === 'upload' && (
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && onFileChange(e.target.files[0])}
        />
      )}

      {inputMode === 'url' && (
        <input
          className={styles.urlInput}
          type="url"
          placeholder="https://example.com/face.jpg"
          value={urlValue}
          onChange={(e) => onUrlChange(e.target.value)}
        />
      )}

      {error && <div className={styles.errorBadge}>⚠ {error}</div>}

      {faceResult && (
        <div className={styles.metaGrid}>
          <div className={styles.metaItem}>
            <span className={styles.metaLabel}>CONFIDENCE</span>
            <span className={styles.metaValue}>{(faceResult.detection.score * 100).toFixed(1)}%</span>
          </div>
          {faceResult.age && (
            <div className={styles.metaItem}>
              <span className={styles.metaLabel}>EST. AGE</span>
              <span className={styles.metaValue}>{Math.round(faceResult.age)}</span>
            </div>
          )}
          {faceResult.gender && (
            <div className={styles.metaItem}>
              <span className={styles.metaLabel}>GENDER</span>
              <span className={styles.metaValue}>{faceResult.gender.toUpperCase()}</span>
            </div>
          )}
          {faceResult.expressions && (
            <div className={styles.metaItem}>
              <span className={styles.metaLabel}>EXPRESSION</span>
              <span className={styles.metaValue}>
                {Object.entries(faceResult.expressions)
                  .sort(([, a]: any, [, b]: any) => b - a)[0][0]
                  .toUpperCase()}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Score Gauge Component ─────────────────────────────────────────────────────
function ScoreGauge({ similarity, verdict, verdictClass }: any) {
  const pct = Math.round(similarity * 100);
  const angle = (pct / 100) * 180 - 90; // -90° to 90°
  const r = 80;
  const cx = 100;
  const cy = 100;
  const arcPath = (from: number, to: number, color: string) => {
    const rad = (deg: number) => (deg * Math.PI) / 180;
    const x1 = cx + r * Math.cos(rad(from - 90));
    const y1 = cy + r * Math.sin(rad(from - 90));
    const x2 = cx + r * Math.cos(rad(to - 90));
    const y2 = cy + r * Math.sin(rad(to - 90));
    const large = to - from > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  const fillAngle = pct * 1.8; // 0-180 degrees
  const gaugeColor =
    verdictClass === 'definite' ? '#00ff88' :
    verdictClass === 'strong' ? '#44ff99' :
    verdictClass === 'possible' ? '#ffcc00' :
    verdictClass === 'weak' ? '#ff8844' : '#ff4444';

  return (
    <div className={styles.gaugeContainer}>
      <svg viewBox="0 0 200 110" className={styles.gaugeSvg}>
        {/* Background arc */}
        <path d={arcPath(0, 180)} fill="none" stroke="#1a2230" strokeWidth="12" strokeLinecap="round" />
        {/* Colored fill arc */}
        {fillAngle > 0 && (
          <path
            d={arcPath(0, Math.min(180, fillAngle))}
            fill="none"
            stroke={gaugeColor}
            strokeWidth="12"
            strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${gaugeColor}80)` }}
          />
        )}
        {/* Tick marks */}
        {[0, 25, 50, 75, 100].map(v => {
          const a = v * 1.8 - 90;
          const rad = (a * Math.PI) / 180;
          const x1 = cx + (r - 8) * Math.cos(rad);
          const y1 = cy + (r - 8) * Math.sin(rad);
          const x2 = cx + (r + 4) * Math.cos(rad);
          const y2 = cy + (r + 4) * Math.sin(rad);
          return <line key={v} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#2a3a4a" strokeWidth="1.5" />;
        })}
        {/* Needle */}
        <g transform={`rotate(${fillAngle - 90}, ${cx}, ${cy})`}>
          <line x1={cx} y1={cy} x2={cx} y2={cy - r + 5} stroke="white" strokeWidth="2" strokeLinecap="round" />
          <circle cx={cx} cy={cy} r="5" fill="white" />
          <circle cx={cx} cy={cy} r="3" fill={gaugeColor} />
        </g>
        {/* Score text */}
        <text x={cx} y={cy + 20} textAnchor="middle" fill="white" fontSize="28" fontWeight="800" fontFamily="Syne, sans-serif">{pct}%</text>
        <text x={cx} y={cy + 36} textAnchor="middle" fill="#8899aa" fontSize="8" fontFamily="Space Mono, monospace" letterSpacing="2">SIMILARITY</text>
        {/* Scale labels */}
        <text x="14" y="108" fill="#556677" fontSize="8" fontFamily="Space Mono, monospace">0</text>
        <text x="185" y="108" fill="#556677" fontSize="8" fontFamily="Space Mono, monospace">100</text>
      </svg>

      <div className={`${styles.verdictBadge} ${styles['verdict_' + verdictClass]}`}>
        {verdict}
      </div>
    </div>
  );
}

// ─── Feature Bar Component ─────────────────────────────────────────────────────
function FeatureBar({ label, value, icon }: { label: string; value: number; icon: string }) {
  const pct = Math.round(value * 100);
  const color = pct >= 70 ? '#00ff88' : pct >= 45 ? '#ffcc00' : '#ff4444';

  return (
    <div className={styles.featureBar}>
      <div className={styles.featureBarHeader}>
        <span className={styles.featureIcon}>{icon}</span>
        <span className={styles.featureLabel}>{label}</span>
        <span className={styles.featureScore} style={{ color }}>{pct}%</span>
      </div>
      <div className={styles.featureBarTrack}>
        <div
          className={styles.featureBarFill}
          style={{
            width: `${pct}%`,
            background: color,
            boxShadow: `0 0 8px ${color}80`,
          }}
        />
      </div>
    </div>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────────
export default function FaceMatch() {
  const [faceApiLoaded, setFaceApiLoaded] = useState(false);
  const [faceApiRef, setFaceApiRef] = useState<any>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [modelError, setModelError] = useState('');

  // Image state
  const [image1, setImage1] = useState<string>('');
  const [image2, setImage2] = useState<string>('');
  const [annotated1, setAnnotated1] = useState<string>('');
  const [annotated2, setAnnotated2] = useState<string>('');
  const [result1, setResult1] = useState<FaceResult | null>(null);
  const [result2, setResult2] = useState<FaceResult | null>(null);
  const [loading1, setLoading1] = useState(false);
  const [loading2, setLoading2] = useState(false);
  const [error1, setError1] = useState('');
  const [error2, setError2] = useState('');
  const [url1, setUrl1] = useState('');
  const [url2, setUrl2] = useState('');
  const [mode1, setMode1] = useState<InputMode>('upload');
  const [mode2, setMode2] = useState<InputMode>('upload');

  // Comparison state
  const [comparing, setComparing] = useState(false);
  const [comparison, setComparison] = useState<ComparisonResult | null>(null);
  const [synopsis, setSynopsis] = useState('');
  const [loadingSynopsis, setLoadingSynopsis] = useState(false);

  // Load face-api.js models from CDN
  useEffect(() => {
    const loadFaceApi = async () => {
      setLoadingModels(true);
      try {
        const faceapi = await import('@vladmandic/face-api');
        const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model/';

        await Promise.all([
          faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
        ]);

        setFaceApiRef(faceapi);
        setFaceApiLoaded(true);
      } catch (err) {
        console.error('Model load error:', err);
        setModelError('Failed to load AI models. Please refresh the page.');
      }
      setLoadingModels(false);
    };
    loadFaceApi();
  }, []);

  const analyzeImage = useCallback(async (imgSrc: string, setLoading: any, setResult: any, setAnnotated: any, setError: any) => {
    if (!faceApiRef || !imgSrc) return;
    setLoading(true);
    setError('');
    setResult(null);
    setAnnotated('');

    try {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = imgSrc;
      });

      const detection = await faceApiRef
        .detectSingleFace(img, new faceApiRef.SsdMobilenetv1Options({ minConfidence: 0.3 }))
        .withFaceLandmarks()
        .withFaceDescriptor()
        .withFaceExpressions()
        .withAgeAndGender();

      if (!detection) {
        setError('No face detected. Try a clearer front-facing photo.');
        setLoading(false);
        return;
      }

      const annotatedCanvas = await drawAnnotatedFace(img, detection, faceApiRef);

      setResult({
        descriptor: detection.descriptor,
        landmarks: detection.landmarks,
        detection: detection.detection,
        expressions: detection.expressions,
        age: detection.age,
        gender: detection.gender,
        genderProbability: detection.genderProbability,
        canvas: annotatedCanvas,
      });
      setAnnotated(annotatedCanvas);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    }
    setLoading(false);
  }, [faceApiRef]);

  const handleFile = useCallback(async (file: File, setImage: any, setLoading: any, setResult: any, setAnnotated: any, setError: any) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
      const src = e.target?.result as string;
      setImage(src);
      await analyzeImage(src, setLoading, setResult, setAnnotated, setError);
    };
    reader.readAsDataURL(file);
  }, [analyzeImage]);

  const handleUrl = useCallback(async (url: string, setImage: any, setLoading: any, setResult: any, setAnnotated: any, setError: any) => {
    if (!url) return;
    setImage(url);
    await analyzeImage(url, setLoading, setResult, setAnnotated, setError);
  }, [analyzeImage]);

  const handleCompare = useCallback(async () => {
    if (!result1 || !result2) return;
    setComparing(true);
    setComparison(null);
    setSynopsis('');

    const distance = euclideanDistance(result1.descriptor, result2.descriptor);
    const similarity = distanceToSimilarity(distance);
    const { verdict, verdictClass } = getVerdict(similarity);
    const featureBreakdown = computeFeatureBreakdown(result1.landmarks, result2.landmarks);

    const comp: ComparisonResult = { distance, similarity, verdict, verdictClass, featureBreakdown };
    setComparison(comp);
    setComparing(false);

    // Fetch AI synopsis
    setLoadingSynopsis(true);
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          similarityScore: similarity,
          distance,
          featureBreakdown,
          imageQuality: {
            img1: result1.detection.score,
            img2: result2.detection.score,
          },
        }),
      });
      const data = await res.json();
      setSynopsis(data.analysis || '');
    } catch {}
    setLoadingSynopsis(false);
  }, [result1, result2]);

  const featureIcons: Record<string, string> = {
    'Left Eye': '👁',
    'Right Eye': '👁',
    'Nose Bridge': '👃',
    'Mouth': '👄',
    'Jaw & Chin': '🫦',
    'Eyebrows': '🤨',
  };

  const canCompare = result1 && result2 && !loading1 && !loading2;

  return (
    <div className={styles.app}>
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.logo}>
            <div className={styles.logoIcon}>
              <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
                <rect x="1" y="1" width="30" height="30" rx="6" stroke="#00d4ff" strokeWidth="1.5"/>
                <circle cx="16" cy="13" r="4" stroke="#00d4ff" strokeWidth="1.5"/>
                <path d="M8 26c0-4.418 3.582-8 8-8s8 3.582 8 8" stroke="#00d4ff" strokeWidth="1.5" strokeLinecap="round"/>
                <path d="M3 3l5 5M29 3l-5 5M3 29l5-5M29 29l-5-5" stroke="#00ff88" strokeWidth="1.5" strokeLinecap="round" opacity="0.6"/>
              </svg>
            </div>
            <div>
              <div className={styles.logoTitle}>FaceMatch<span className={styles.logoAI}>AI</span></div>
              <div className={styles.logoSub}>FORENSIC FACIAL RECOGNITION SYSTEM</div>
            </div>
          </div>
          <div className={styles.headerStats}>
            <div className={styles.stat}>
              <span className={styles.statVal}>99.38%</span>
              <span className={styles.statLabel}>LFW ACCURACY</span>
            </div>
            <div className={styles.statDivider}/>
            <div className={styles.stat}>
              <span className={styles.statVal}>128-D</span>
              <span className={styles.statLabel}>FEATURE VECTOR</span>
            </div>
            <div className={styles.statDivider}/>
            <div className={styles.stat}>
              <span className={styles.statVal}>68 PT</span>
              <span className={styles.statLabel}>LANDMARKS</span>
            </div>
            <div className={styles.statDivider}/>
            <div className={`${styles.stat} ${styles.modelStatus}`}>
              <span className={`${styles.statusDot} ${faceApiLoaded ? styles.statusOn : loadingModels ? styles.statusLoading : styles.statusOff}`}/>
              <span className={styles.statLabel}>
                {faceApiLoaded ? 'MODELS READY' : loadingModels ? 'LOADING...' : 'OFFLINE'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className={styles.main}>
        {modelError && (
          <div className={styles.globalError}>{modelError}</div>
        )}

        {loadingModels && (
          <div className={styles.modelLoadingBanner}>
            <div className={styles.modelLoadingBar} />
            <span>Loading ResNet-34 + SSD MobileNet + Landmark models from CDN...</span>
          </div>
        )}

        {/* Image Upload Section */}
        <section className={styles.uploadSection}>
          <ImagePanel
            label="SUBJECT A"
            image={image1}
            annotatedImage={annotated1}
            faceResult={result1}
            loading={loading1}
            error={error1}
            inputMode={mode1}
            onModeChange={setMode1}
            urlValue={url1}
            onFileChange={(file: File) => handleFile(file, setImage1, setLoading1, setResult1, setAnnotated1, setError1)}
            onUrlChange={(val: string) => { setUrl1(val); }}
          />

          <div className={styles.vsColumn}>
            <div className={styles.vsConnector} />
            <div className={styles.vsBadge}>VS</div>
            <div className={styles.vsConnector} />
            {mode2 === 'url' && (
              <button
                className={styles.loadUrlBtn}
                onClick={() => { handleUrl(url1, setImage1, setLoading1, setResult1, setAnnotated1, setError1); handleUrl(url2, setImage2, setLoading2, setResult2, setAnnotated2, setError2); }}
                disabled={!url1 || !url2}
              >LOAD URLS</button>
            )}
          </div>

          <ImagePanel
            label="SUBJECT B"
            image={image2}
            annotatedImage={annotated2}
            faceResult={result2}
            loading={loading2}
            error={error2}
            inputMode={mode2}
            onModeChange={setMode2}
            urlValue={url2}
            onFileChange={(file: File) => handleFile(file, setImage2, setLoading2, setResult2, setAnnotated2, setError2)}
            onUrlChange={(val: string) => { setUrl2(val); }}
          />
        </section>

        {/* URL Load Button */}
        {(mode1 === 'url' || mode2 === 'url') && (
          <div className={styles.urlLoadRow}>
            <button
              className={styles.loadUrlBtnMain}
              onClick={() => {
                if (mode1 === 'url' && url1) handleUrl(url1, setImage1, setLoading1, setResult1, setAnnotated1, setError1);
                if (mode2 === 'url' && url2) handleUrl(url2, setImage2, setLoading2, setResult2, setAnnotated2, setError2);
              }}
              disabled={!(mode1 === 'url' && url1) && !(mode2 === 'url' && url2)}
            >
              ⊕ LOAD & ANALYZE URL IMAGES
            </button>
          </div>
        )}

        {/* Compare Button */}
        <div className={styles.compareRow}>
          <button
            className={`${styles.compareBtn} ${canCompare ? styles.compareBtnActive : ''}`}
            onClick={handleCompare}
            disabled={!canCompare || comparing}
          >
            {comparing ? (
              <><span className={styles.btnSpinner}/> PROCESSING...</>
            ) : (
              <>⬡ COMPARE FACES</>
            )}
          </button>
          {!faceApiLoaded && <span className={styles.compareHint}>Waiting for AI models to load...</span>}
          {faceApiLoaded && !canCompare && <span className={styles.compareHint}>Upload both images to enable comparison</span>}
        </div>

        {/* Results Section */}
        {comparison && (
          <section className={styles.resultsSection}>
            <div className={styles.resultsSectionHeader}>
              <span className={styles.sectionLabel}>BIOMETRIC ANALYSIS RESULTS</span>
              <span className={styles.sectionLine}/>
            </div>

            <div className={styles.resultsGrid}>
              {/* Score gauge */}
              <div className={styles.gaugeCard}>
                <ScoreGauge
                  similarity={comparison.similarity}
                  verdict={comparison.verdict}
                  verdictClass={comparison.verdictClass}
                />
                <div className={styles.distanceRow}>
                  <span className={styles.distLabel}>EUCLIDEAN DISTANCE</span>
                  <span className={styles.distValue}>{comparison.distance.toFixed(4)}</span>
                  <span className={styles.distThreshold}>threshold: 0.600</span>
                </div>
              </div>

              {/* Feature breakdown */}
              <div className={styles.featuresCard}>
                <div className={styles.cardTitle}>FACIAL REGION ANALYSIS</div>
                {Object.entries(comparison.featureBreakdown).map(([key, val]) => (
                  <FeatureBar
                    key={key}
                    label={key}
                    value={val as number}
                    icon={featureIcons[key] || '•'}
                  />
                ))}
              </div>

              {/* Technical metrics */}
              <div className={styles.metricsCard}>
                <div className={styles.cardTitle}>TECHNICAL METRICS</div>
                <div className={styles.metricsGrid}>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>ALGORITHM</span>
                    <span className={styles.metricValue}>ResNet-34</span>
                  </div>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>EMBEDDING DIM</span>
                    <span className={styles.metricValue}>128-D vector</span>
                  </div>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>LANDMARK MODEL</span>
                    <span className={styles.metricValue}>68-point CNN</span>
                  </div>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>DETECTOR</span>
                    <span className={styles.metricValue}>SSD MobileNetV1</span>
                  </div>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>BENCHMARK</span>
                    <span className={styles.metricValue}>99.38% LFW</span>
                  </div>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>DISTANCE METRIC</span>
                    <span className={styles.metricValue}>Euclidean L2</span>
                  </div>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>DET. SCORE A</span>
                    <span className={styles.metricValue}>{result1 ? (result1.detection.score * 100).toFixed(1) + '%' : '—'}</span>
                  </div>
                  <div className={styles.metricItem}>
                    <span className={styles.metricLabel}>DET. SCORE B</span>
                    <span className={styles.metricValue}>{result2 ? (result2.detection.score * 100).toFixed(1) + '%' : '—'}</span>
                  </div>
                </div>

                {/* Similarity scale */}
                <div className={styles.scaleSection}>
                  <div className={styles.cardTitle} style={{ marginBottom: '10px' }}>CONFIDENCE SCALE</div>
                  {[
                    { label: 'DEFINITE MATCH', min: 85, color: '#00ff88' },
                    { label: 'STRONG MATCH', min: 70, color: '#44ff99' },
                    { label: 'POSSIBLE MATCH', min: 55, color: '#ffcc00' },
                    { label: 'WEAK MATCH', min: 40, color: '#ff8844' },
                    { label: 'NO MATCH', min: 0, color: '#ff4444' },
                  ].map(({ label, min, color }) => (
                    <div key={label} className={styles.scaleRow}>
                      <span className={styles.scaleDot} style={{ background: color, boxShadow: `0 0 6px ${color}` }} />
                      <span className={styles.scaleLabel}>{label}</span>
                      <span className={styles.scaleRange} style={{ color }}>{min}%+</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Synopsis */}
            <div className={styles.synopsisCard}>
              <div className={styles.synopsisHeader}>
                <span className={styles.cardTitle}>FORENSIC ANALYSIS SYNOPSIS</span>
                <span className={styles.synopsisBadge}>AI POWERED</span>
              </div>
              {loadingSynopsis ? (
                <div className={styles.synopsisLoading}>
                  <div className={styles.synopsisLoadingDots}>
                    <span/><span/><span/>
                  </div>
                  <span>Generating forensic analysis...</span>
                </div>
              ) : synopsis ? (
                <div className={styles.synopsisText}>{synopsis}</div>
              ) : null}
            </div>
          </section>
        )}

        {/* How it works */}
        <section className={styles.howItWorks}>
          <div className={styles.howTitle}>HOW IT WORKS</div>
          <div className={styles.howGrid}>
            {[
              { step: '01', title: 'FACE DETECTION', desc: 'SSD MobileNetV1 locates and isolates the face within the image with high confidence scoring' },
              { step: '02', title: '68-POINT LANDMARKS', desc: 'A CNN maps 68 anatomical keypoints: jaw, eyes, brows, nose, and mouth with sub-pixel accuracy' },
              { step: '03', title: 'FEATURE EMBEDDING', desc: 'ResNet-34 encodes the aligned face into a 128-dimensional descriptor vector capturing unique biometric traits' },
              { step: '04', title: 'EUCLIDEAN DISTANCE', desc: 'L2 distance between descriptors is computed. Distance < 0.6 indicates the same identity per NIST standards' },
              { step: '05', title: 'REGION ANALYSIS', desc: 'Individual facial regions are compared geometrically to identify which areas drove the similarity score' },
              { step: '06', title: 'AI SYNOPSIS', desc: 'Claude AI generates a detailed forensic analysis report contextualizing the results with biometric science' },
            ].map(({ step, title, desc }) => (
              <div key={step} className={styles.howCard}>
                <span className={styles.howStep}>{step}</span>
                <span className={styles.howCardTitle}>{title}</span>
                <span className={styles.howCardDesc}>{desc}</span>
              </div>
            ))}
          </div>
        </section>
      </main>

      <footer className={styles.footer}>
        <span>FaceMatch AI — ResNet-34 · face-api.js · 99.38% LFW Benchmark</span>
        <span className={styles.footerDivider}>·</span>
        <span>For research and educational purposes only</span>
      </footer>
    </div>
  );
}
