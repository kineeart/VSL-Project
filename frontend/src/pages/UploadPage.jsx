import React, { useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';

export default function UploadPage() {
  const { t } = useTranslation();
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [debugInfo, setDebugInfo] = useState(null);
  const [error, setError] = useState('');
  const [videoUrl, setVideoUrl] = useState('');

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (f) {
      setFile(f);
      setPredictions([]);
      setDebugInfo(null);
      setError('');
      setVideoUrl(URL.createObjectURL(f));
    }
  };

  const predict = async () => {
    if (!file) return;
    setLoading(true);
    setError('');
    setPredictions([]);
    setDebugInfo(null);

    const form = new FormData();
    form.append('file', file);

    try {
      const res = await fetch('/api/predict/video', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) { setError(data.error || 'Error'); return; }
      setPredictions(data.predictions || []);
      setDebugInfo(data.debug || null);
    } catch {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="card">
        <h2>{t('upload.title')}</h2>
        <div className="actions">
          <button className="btn" onClick={() => fileRef.current?.click()}>
            {t('upload.select')}
          </button>
          <input ref={fileRef} type="file" accept="video/*" onChange={handleFile} hidden aria-label={t('upload.select')} />
          <button className="btn btn-primary" onClick={predict} disabled={!file || loading}>
            {loading ? t('upload.processing') : t('upload.predict')}
          </button>
        </div>
        {file && <p style={{ marginTop: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{file.name}</p>}
        {videoUrl && (
          <div className="video-container" style={{ marginTop: '1rem' }}>
            <video src={videoUrl} controls playsInline aria-label={file?.name} />
          </div>
        )}
        {error && <div className="alert alert-error">{error}</div>}
      </div>

      {predictions.length > 0 && (
        <div className="card">
          <h2>{t('upload.result')}</h2>
          <div className="predictions">
            {predictions.map((p, i) => (
              <div className="prediction-item" key={i}>
                <span className="prediction-label">{p.label}</span>
                <div className="prediction-bar">
                  <div className="prediction-fill" style={{ width: `${(p.confidence * 100).toFixed(0)}%` }} />
                </div>
                <span className="prediction-pct">{(p.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>

          {debugInfo && (
            <div className="debug-panel">
              <div className="debug-title">Debug (raw vs smooth)</div>
              <div className="debug-row">reason: {debugInfo.reason}</div>
              <div className="debug-row">margin: {(debugInfo.margin ?? 0).toFixed(3)}</div>
              <div className="debug-row">raw: {(debugInfo.raw_topk || []).map((x) => `${x.label}:${(x.confidence * 100).toFixed(1)}%`).join(' | ')}</div>
              <div className="debug-row">smooth: {(debugInfo.smooth_topk || []).map((x) => `${x.label}:${(x.confidence * 100).toFixed(1)}%`).join(' | ')}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
