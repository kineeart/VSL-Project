import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

export default function ContinuousPage() {
  const { t } = useTranslation();
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [committedText, setCommittedText] = useState('');
  const [liveText, setLiveText] = useState('');
  const [committedTokens, setCommittedTokens] = useState([]);
  const [liveTokens, setLiveTokens] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [debugInfo, setDebugInfo] = useState(null);
  const [error, setError] = useState('');

  const stop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
  }, []);

  useEffect(() => () => stop(), [stop]);

  const resetPhrase = () => {
    setCommittedText('');
    setLiveText('');
    setCommittedTokens([]);
    setLiveTokens([]);
    setPredictions([]);
    setDebugInfo(null);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'reset' }));
    }
  };

  const start = async () => {
    setError('');
    resetPhrase();

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      setError('Cannot access camera');
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/continuous`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.predictions) setPredictions(data.predictions);
      if (data.continuous) {
        setCommittedTokens(data.continuous.committed_tokens || []);
        setLiveTokens(data.continuous.live_tokens || []);
        setCommittedText(data.continuous.committed_text || '');
        setLiveText(data.continuous.live_text || '');
      }
      if (data.debug) setDebugInfo(data.debug);
    };
    ws.onerror = () => setError('WebSocket error');

    ws.onopen = () => {
      setRunning(true);
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');

      intervalRef.current = setInterval(() => {
        if (videoRef.current && ws.readyState === WebSocket.OPEN) {
          ctx.drawImage(videoRef.current, 0, 0, 640, 480);
          const b64 = canvas.toDataURL('image/jpeg', 0.7);
          ws.send(JSON.stringify({ type: 'frame', data: b64 }));
        }
      }, 200);
    };
  };

  return (
    <div>
      <div className="card">
        <h2>{t('continuous.title')}</h2>
        <div className="video-container">
          <video ref={videoRef} muted playsInline aria-label={t('continuous.title')} />
        </div>
        <div className="actions">
          {!running ? (
            <button className="btn btn-primary" onClick={start}>{t('continuous.start')}</button>
          ) : (
            <button className="btn btn-danger" onClick={stop}>{t('continuous.stop')}</button>
          )}
          <button className="btn" onClick={resetPhrase}>{t('continuous.reset')}</button>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
      </div>

      <div className="card">
        <div className="split-grid">
          <div className="split-card">
            <div className="label">{t('continuous.committed')}</div>
            <div className="value">{committedText || t('continuous.waiting')}</div>
          </div>
          <div className="split-card">
            <div className="label">{t('continuous.live')}</div>
            <div className="value">{liveText || t('continuous.waiting')}</div>
          </div>
        </div>

        <div className="phrase-box">
          <div className="phrase-title">{t('continuous.committed')}</div>
          <div className="phrase-live">
            {committedText || <span className="phrase-muted">{t('continuous.waiting')}</span>}
          </div>
          <div className="token-strip">
            {committedTokens.length ? committedTokens.map((token, index) => (
              <span className="token-chip" key={`${token}-${index}`}>{token}</span>
            )) : <span className="token-chip muted">---</span>}
          </div>
        </div>

        <div className="phrase-box">
          <div className="phrase-title">{t('continuous.live')}</div>
          <div className="phrase-live">
            {liveText || <span className="phrase-muted">{t('continuous.waiting')}</span>}
          </div>
          <div className="token-strip">
            {liveTokens.length ? liveTokens.map((token, index) => (
              <span className="token-chip" key={`${token}-${index}`}>{token}</span>
            )) : <span className="token-chip muted">---</span>}
          </div>
        </div>

        {predictions.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>{t('camera.waiting')}</p>
        ) : (
          <div className="predictions">
            {predictions.map((p, index) => (
              <div className="prediction-item" key={`${p.label}-${index}`}>
                <span className="prediction-label">{p.label}</span>
                <div className="prediction-bar">
                  <div className="prediction-fill" style={{ width: `${(p.confidence * 100).toFixed(0)}%` }} />
                </div>
                <span className="prediction-pct">{(p.confidence * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        )}

        {debugInfo && (
          <div className="debug-panel">
            <div className="debug-title">Debug (raw vs smooth)</div>
            <div className="debug-row">reason: {debugInfo.reason}</div>
            <div className="debug-row">margin: {(debugInfo.margin ?? 0).toFixed(3)}</div>
            <div className="debug-row">raw: {(debugInfo.raw_topk || []).map((item) => `${item.label}:${(item.confidence * 100).toFixed(1)}%`).join(' | ')}</div>
            <div className="debug-row">smooth: {(debugInfo.smooth_topk || []).map((item) => `${item.label}:${(item.confidence * 100).toFixed(1)}%`).join(' | ')}</div>
          </div>
        )}
      </div>
    </div>
  );
}