import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export default function StatusPage() {
  const { t } = useTranslation();
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/status')
      .then(r => r.json())
      .then(setStatus)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="card"><p>Loading...</p></div>;

  return (
    <div className="card">
      <h2>{t('status.title')}</h2>
      <div className="status-grid">
        <div className="status-item">
          <div className="value" style={{ color: status?.model_loaded ? 'var(--success)' : 'var(--danger)' }}>
            {status?.model_loaded ? t('status.yes') : t('status.no')}
          </div>
          <div className="label">{t('status.modelLoaded')}</div>
        </div>
        <div className="status-item">
          <div className="value">{status?.num_classes || 0}</div>
          <div className="label">{t('status.numClasses')}</div>
        </div>
      </div>
      {status?.labels?.length > 0 && (
        <div style={{ marginTop: '1rem', maxHeight: '300px', overflow: 'auto' }}>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
            Labels ({status.labels.length}):
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
            {status.labels.map((l, i) => (
              <span key={i} style={{
                padding: '0.2rem 0.5rem',
                background: 'var(--bg)',
                borderRadius: '4px',
                fontSize: '0.8rem',
                border: '1px solid var(--border)'
              }}>{l}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
