import React, { useRef, useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export default function CameraPage() {
  const { t } = useTranslation();
  const videoRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const [running, setRunning] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState('');

  const stop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = null;
    if (wsRef.current) { wsRef.current.close(); wsRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
    setRunning(false);
  }, []);

  useEffect(() => () => stop(), [stop]);

  const start = async () => {
    setError('');
    setPredictions([]);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
    } catch {
      setError('Cannot access camera');
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/predict`);
    wsRef.current = ws;

    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.predictions) setPredictions(data.predictions);
    };
    ws.onerror = () => setError('WebSocket error');
    ws.onclose = () => {};

    ws.onopen = () => {
      setRunning(true);
      const canvas = document.createElement('canvas');
      canvas.width = 640; canvas.height = 480;
      const ctx = canvas.getContext('2d');

      intervalRef.current = setInterval(() => {
        if (videoRef.current && ws.readyState === WebSocket.OPEN) {
          ctx.drawImage(videoRef.current, 0, 0, 640, 480);
          const b64 = canvas.toDataURL('image/jpeg', 0.7);
          ws.send(JSON.stringify({ type: 'frame', data: b64 }));
        }
      }, 200); // 5 fps
    };
  };

  return (
    <div>
      <div className="card">
        <h2>{t('camera.title')}</h2>
        <div className="video-container">
          <video ref={videoRef} muted playsInline aria-label={t('camera.title')} />
        </div>
        <div className="actions">
          {!running ? (
            <button className="btn btn-primary" onClick={start}>{t('camera.start')}</button>
          ) : (
            <button className="btn btn-danger" onClick={stop}>{t('camera.stop')}</button>
          )}
        </div>
        {error && <div className="alert alert-error">{error}</div>}
      </div>

      <div className="card">
        {predictions.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>{t('camera.waiting')}</p>
        ) : (
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
        )}
      </div>
    </div>
  );
}
