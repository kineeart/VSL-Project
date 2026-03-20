import React, { useRef, useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export default function TrainingPage() {
  const { t } = useTranslation();
  const videoRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const [recording, setRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState('');
  const [label, setLabel] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [customList, setCustomList] = useState({});
  const [maxVideos, setMaxVideos] = useState(200);
  const [trainStatus, setTrainStatus] = useState(''); // '' | 'running' | 'done' | 'error'
  const [trainResult, setTrainResult] = useState(null);

  const loadCustomList = useCallback(async () => {
    try {
      const res = await fetch('/api/train/custom/list');
      const data = await res.json();
      setCustomList(data);
    } catch {}
  }, []);

  useEffect(() => { loadCustomList(); }, [loadCustomList]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const startRecording = async () => {
    setSaved(false);
    setRecordedBlob(null);
    setRecordedUrl('');
    chunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
    } catch { return; }

    const recorder = new MediaRecorder(streamRef.current, { mimeType: 'video/webm' });
    recorderRef.current = recorder;
    recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      setRecordedBlob(blob);
      setRecordedUrl(URL.createObjectURL(blob));
      stopCamera();
    };
    recorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
    setRecording(false);
  };

  const saveSample = async () => {
    if (!recordedBlob || !label.trim()) return;
    setSaving(true);
    setSaved(false);

    const form = new FormData();
    form.append('file', recordedBlob, `custom_${Date.now()}.webm`);
    form.append('label', label.trim());

    try {
      const res = await fetch('/api/train/custom', { method: 'POST', body: form });
      if (res.ok) {
        setSaved(true);
        setRecordedBlob(null);
        setRecordedUrl('');
        setLabel('');
        loadCustomList();
      }
    } catch {}
    setSaving(false);
  };

  const startTraining = async () => {
    setTrainStatus('running');
    setTrainResult(null);
    try {
      const res = await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_videos: maxVideos })
      });
      const data = await res.json();
      if (res.ok) { setTrainStatus('done'); setTrainResult(data); }
      else { setTrainStatus('error'); setTrainResult(data); }
    } catch {
      setTrainStatus('error');
    }
  };

  const customCount = Object.keys(customList).length;

  return (
    <div>
      <div className="card">
        <h2>{t('training.title')}</h2>

        <div className="video-container">
          <video ref={videoRef} muted playsInline aria-label={t('training.record')} />
        </div>

        {recordedUrl && (
          <div className="video-container" style={{ marginTop: '0.5rem' }}>
            <video src={recordedUrl} controls playsInline aria-label={t('training.label')} />
          </div>
        )}

        <div className="actions">
          {!recording ? (
            <button className="btn btn-primary" onClick={startRecording}>{t('training.record')}</button>
          ) : (
            <button className="btn btn-danger" onClick={stopRecording}>{t('training.stopRecord')}</button>
          )}
        </div>

        {recordedBlob && (
          <div style={{ marginTop: '1rem' }}>
            <div className="form-group">
              <label htmlFor="label-input">{t('training.label')}</label>
              <input
                id="label-input"
                type="text"
                value={label}
                onChange={e => setLabel(e.target.value)}
                placeholder={t('training.labelPlaceholder')}
              />
            </div>
            <button className="btn btn-success" onClick={saveSample} disabled={saving || !label.trim()}>
              {saving ? '...' : t('training.save')}
            </button>
          </div>
        )}

        {saved && <div className="alert alert-success" style={{ marginTop: '0.5rem' }}>{t('training.saved')}</div>}
      </div>

      <div className="card">
        <h2>{t('training.trainModel')}</h2>
        <p style={{ marginBottom: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          {t('training.customSamples')}: {customCount}
        </p>
        <div className="form-group">
          <label htmlFor="max-videos">{t('training.maxVideos')}</label>
          <input
            id="max-videos"
            type="number"
            min={10}
            max={5000}
            value={maxVideos}
            onChange={e => setMaxVideos(Number(e.target.value))}
          />
        </div>
        <button
          className="btn btn-primary"
          onClick={startTraining}
          disabled={trainStatus === 'running'}
        >
          {trainStatus === 'running' ? t('training.trainRunning') : t('training.trainStart')}
        </button>

        {trainStatus === 'done' && (
          <div className="alert alert-success" style={{ marginTop: '0.75rem' }}>
            {t('training.trainDone')}
            {trainResult && ` (${trainResult.num_classes} classes, ${trainResult.num_samples} samples)`}
          </div>
        )}
        {trainStatus === 'error' && (
          <div className="alert alert-error" style={{ marginTop: '0.75rem' }}>
            {trainResult?.error || 'Training failed'}
          </div>
        )}
      </div>
    </div>
  );
}
