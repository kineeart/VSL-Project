import React, { useRef, useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

const PREP_SECONDS = 2;

export default function TrainingPage() {
  const { t } = useTranslation();
  const videoRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const abortRef = useRef(false);

  const [recording, setRecording] = useState(false);
  const [recordedUrl, setRecordedUrl] = useState('');
  const [label, setLabel] = useState('');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [customList, setCustomList] = useState({});
  const [maxVideos, setMaxVideos] = useState(200);
  const [targetVideos, setTargetVideos] = useState(5);
  const [durationSeconds, setDurationSeconds] = useState(3);
  const [countdown, setCountdown] = useState(0);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0, label: '' });
  const [batchStatus, setBatchStatus] = useState('idle'); // idle | running | done | error
  const [batchMessage, setBatchMessage] = useState('');
  const [downloadingLabel, setDownloadingLabel] = useState('');
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

  useEffect(() => {
    return () => {
      abortRef.current = true;
      if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    };
  }, []);

  useEffect(() => {
    return () => {
      if (recordedUrl) URL.revokeObjectURL(recordedUrl);
    };
  }, [recordedUrl]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const sleep = useCallback((ms) => new Promise(resolve => setTimeout(resolve, ms)), []);

  const waitForCountdown = useCallback(async (seconds) => {
    for (let remaining = seconds; remaining > 0; remaining -= 1) {
      if (abortRef.current) return false;
      setCountdown(remaining);
      await sleep(1000);
    }
    setCountdown(0);
    return !abortRef.current;
  }, [sleep]);

  const recordOneSample = useCallback(async (seconds) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      return null;
    }

    const recorder = new MediaRecorder(streamRef.current, { mimeType: 'video/webm' });
    recorderRef.current = recorder;
    chunksRef.current = [];

    const blobPromise = new Promise((resolve) => {
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        setRecordedUrl(prevUrl => {
          if (prevUrl) URL.revokeObjectURL(prevUrl);
          return URL.createObjectURL(blob);
        });
        stopCamera();
        resolve(blob);
      };
    });

    recorder.start();
    setRecording(true);
    await sleep(seconds * 1000);
    if (recorder.state !== 'inactive') recorder.stop();
    const blob = await blobPromise;
    setRecording(false);
    return blob;
  }, [sleep, stopCamera]);

  const uploadSample = useCallback(async (blob, sampleLabel) => {
    const form = new FormData();
    form.append('file', blob, `custom_${Date.now()}.webm`);
    form.append('label', sampleLabel);

    const res = await fetch('/api/train/custom', { method: 'POST', body: form });
    return res.ok;
  }, []);

  const checkBackend = useCallback(async () => {
    try {
      const res = await fetch('/api/status');
      return res.ok;
    } catch {
      return false;
    }
  }, []);

  const startRecording = async () => {
    const cleanLabel = label.trim();
    if (!cleanLabel) return;

    setSaved(false);
    setBatchMessage('');
    setBatchStatus('running');
    setBatchProgress({ current: 1, total: 1, label: cleanLabel });
    abortRef.current = false;

    const prepared = await waitForCountdown(PREP_SECONDS);
    if (!prepared) {
      setBatchStatus('idle');
      return;
    }

    const blob = await recordOneSample(durationSeconds);
    if (!blob) {
      setBatchStatus('error');
      setBatchMessage(t('training.recordFailed'));
      return;
    }

    const ok = await uploadSample(blob, cleanLabel);
    if (!ok) {
      setBatchStatus('error');
      setBatchMessage(t('training.saveFailed', { current: 1 }));
      return;
    }

    setSaved(true);
    setBatchStatus('done');
    setBatchMessage(t('training.batchDone'));
    loadCustomList();
  };

  const stopRecording = () => {
    abortRef.current = true;
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
    setRecording(false);
    setCountdown(0);
    if (batchStatus === 'running') setBatchStatus('idle');
  };

  const startAutoRecording = async () => {
    const cleanLabel = label.trim();
    const total = Math.max(1, Number(targetVideos) || 1);

    if (!cleanLabel || batchStatus === 'running') return;

    const backendReady = await checkBackend();
    if (!backendReady) {
      setBatchStatus('error');
      setBatchMessage(t('training.backendOffline'));
      return;
    }

    abortRef.current = false;
    setSaved(false);
    setSaving(false);
    setBatchMessage('');
    setBatchStatus('running');
    setBatchProgress({ current: 0, total, label: cleanLabel });

    let failed = false;
    for (let index = 0; index < total; index += 1) {
      if (abortRef.current) break;

      setBatchProgress({ current: index + 1, total, label: cleanLabel });
      setBatchMessage(t('training.preparing', { seconds: PREP_SECONDS }));

      const prepared = await waitForCountdown(PREP_SECONDS);
      if (!prepared) break;

      const blob = await recordOneSample(durationSeconds);
      if (!blob) {
        failed = true;
        setBatchMessage(t('training.recordFailed'));
        break;
      }

      const ok = await uploadSample(blob, cleanLabel);
      if (!ok) {
        failed = true;
        setBatchMessage(t('training.saveFailed', { current: index + 1 }));
        break;
      }
    }

    setCountdown(0);
    setRecording(false);
    setSaving(false);

    if (!failed && !abortRef.current) {
      setBatchStatus('done');
      setBatchMessage(t('training.batchDone'));
      setSaved(true);
      loadCustomList();
    } else if (failed) {
      setBatchStatus('error');
    } else {
      setBatchStatus('idle');
    }
  };

  const downloadLabel = async (targetLabel) => {
    setDownloadingLabel(targetLabel);
    try {
      const res = await fetch(`/api/train/custom/download/${encodeURIComponent(targetLabel)}`);
      if (!res.ok) throw new Error('download failed');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `custom_videos_${targetLabel}.zip`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch {
      setBatchMessage(t('training.downloadFailed'));
    } finally {
      setDownloadingLabel('');
    }
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
  const labelStats = Object.entries(customList).reduce((acc, [, itemLabel]) => {
    acc[itemLabel] = (acc[itemLabel] || 0) + 1;
    return acc;
  }, {});
  const labelEntries = Object.entries(labelStats).sort((a, b) => a[0].localeCompare(b[0]));

  return (
    <div>
      <div className="card">
        <h2>{t('training.title')}</h2>

        <p style={{ marginBottom: '0.75rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          {t('training.chooseLabel')}
        </p>

        <div className="form-group">
          <label htmlFor="label-input">{t('training.label')}</label>
          <input
            id="label-input"
            type="text"
            value={label}
            onChange={e => setLabel(e.target.value)}
            placeholder={t('training.labelPlaceholder')}
            list="label-options"
          />
          <datalist id="label-options">
            {labelEntries.map(([itemLabel]) => (
              <option key={itemLabel} value={itemLabel} />
            ))}
          </datalist>
        </div>

        <div className="form-group">
          <label htmlFor="target-videos">{t('training.quantity')}</label>
          <input
            id="target-videos"
            type="number"
            min={1}
            max={200}
            value={targetVideos}
            onChange={e => setTargetVideos(Number(e.target.value))}
          />
        </div>

        <div className="form-group">
          <label htmlFor="duration-seconds">{t('training.duration')}: {durationSeconds}s</label>
          <input
            id="duration-seconds"
            type="range"
            min={2}
            max={10}
            step={1}
            value={durationSeconds}
            onChange={e => setDurationSeconds(Number(e.target.value))}
          />
        </div>

        <p style={{ marginTop: '0.25rem', marginBottom: '0.75rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          {t('training.autoSaveNote')}
        </p>

        <div className="video-container">
          <video ref={videoRef} muted playsInline aria-label={t('training.record')} />
        </div>

        {countdown > 0 && (
          <div className="alert alert-success" style={{ marginTop: '0.75rem' }}>
            {t('training.preparing', { seconds: countdown })}
          </div>
        )}

        {batchProgress.total > 0 && batchStatus === 'running' && (
          <div className="alert" style={{ marginTop: '0.75rem' }}>
            {t('training.batchProgress', {
              current: batchProgress.current,
              total: batchProgress.total,
              label: batchProgress.label
            })}
          </div>
        )}

        {batchMessage && (
          <div className={batchStatus === 'error' ? 'alert alert-error' : 'alert alert-success'} style={{ marginTop: '0.75rem' }}>
            {batchMessage}
          </div>
        )}

        {recordedUrl && (
          <div className="video-container" style={{ marginTop: '0.5rem' }}>
            <video src={recordedUrl} controls playsInline aria-label={t('training.label')} />
          </div>
        )}

        <div className="actions">
          {recording || batchStatus === 'running' ? (
            <button className="btn btn-danger" onClick={stopRecording}>{t('training.stopRecord')}</button>
          ) : (
            <button className="btn btn-primary" onClick={startAutoRecording} disabled={batchStatus === 'running' || !label.trim()}>
              {t('training.autoRecord')}
            </button>
          )}
        </div>

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

      {labelEntries.length > 0 && (
        <div className="card">
          <h2>{t('training.existingLabels')}</h2>
          <div style={{ display: 'grid', gap: '0.75rem' }}>
            {labelEntries.map(([itemLabel, count]) => (
              <div key={itemLabel} className="form-group" style={{ marginBottom: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', alignItems: 'center' }}>
                  <div>
                    <strong>{itemLabel}</strong>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      {count} video(s)
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button className="btn btn-secondary" onClick={() => setLabel(itemLabel)}>
                      {t('training.label')}
                    </button>
                    <button
                      className="btn btn-primary"
                      onClick={() => downloadLabel(itemLabel)}
                      disabled={downloadingLabel === itemLabel}
                    >
                      {downloadingLabel === itemLabel ? '...' : t('training.downloadLabel')}
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
