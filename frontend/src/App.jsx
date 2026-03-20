import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import CameraPage from './pages/CameraPage';
import UploadPage from './pages/UploadPage';
import TrainingPage from './pages/TrainingPage';
import StatusPage from './pages/StatusPage';

const PAGES = ['camera', 'upload', 'training', 'status'];

export default function App() {
  const { t, i18n } = useTranslation();
  const [page, setPage] = useState('camera');
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');
  const toggleLang = () => i18n.changeLanguage(i18n.language === 'vi' ? 'en' : 'vi');

  return (
    <div className="app">
      <header className="header">
        <h1>{t('appName')}</h1>
        <div className="header-controls">
          <button className="btn btn-sm" onClick={toggleLang}>
            {i18n.language === 'vi' ? t('lang.en') : t('lang.vi')}
          </button>
          <button className="btn btn-sm" onClick={toggleTheme}>
            {theme === 'light' ? t('theme.dark') : t('theme.light')}
          </button>
        </div>
      </header>

      <nav className="nav" role="navigation" aria-label={t('appName')}>
        {PAGES.map(p => (
          <button
            key={p}
            className={page === p ? 'active' : ''}
            onClick={() => setPage(p)}
            aria-current={page === p ? 'page' : undefined}
          >
            {t(`nav.${p}`)}
          </button>
        ))}
      </nav>

      <main className="content">
        {page === 'camera' && <CameraPage />}
        {page === 'upload' && <UploadPage />}
        {page === 'training' && <TrainingPage />}
        {page === 'status' && <StatusPage />}
      </main>
    </div>
  );
}
