/**
 * web/src/App.tsx
 * Primary Application Shell with Hash Routing and Accessible Navigation
 */

import React, { useState, useEffect } from 'react';
import {
  HashRouter,
  Routes,
  Route,
  NavLink,
  Navigate,
  useLocation,
} from 'react-router-dom';
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query';
import { checkHealth } from './api/health';
import { ScanRoute } from './routes/ScanRoute';
import { ResultRoute } from './routes/ResultRoute';
import { HistoryRoute } from './routes/HistoryRoute';
import { AboutRoute } from './routes/AboutRoute';
import { SystemRoute } from './routes/SystemRoute';
import styles from './App.module.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
    },
  },
});

const ScrollAndFocusManager: React.FC<{ announce: (msg: string) => void }> = ({
  announce,
}) => {
  const location = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Focus management on route change
    const mainContent = document.getElementById('main-content');
    if (mainContent) {
      mainContent.focus();
    }

    // Dynamic document title
    const path = location.pathname;
    if (path.startsWith('/result')) {
      document.title = 'Pathology Report — LeafScan V2';
      announce('Diagnostic pathology report loaded.');
    } else if (path === '/history') {
      document.title = 'Diagnostic History — LeafScan V2';
      announce('Diagnostic history log loaded.');
    } else if (path === '/about') {
      document.title = 'About & Methodology — LeafScan V2';
      announce('About technical specifications loaded.');
    } else if (path === '/system') {
      document.title = 'Backend System Telemetry — LeafScan V2';
      announce('Backend system telemetry loaded.');
    } else {
      document.title = 'LeafScan V2 — Foliar Disease Detection';
      announce('Foliar diagnosis scan workspace ready.');
    }
  }, [location, announce]);

  return null;
};

export const AppContent: React.FC = () => {
  const [ariaAnnouncement, setAriaAnnouncement] = useState('');

  // Background health polling to inform foliar workspace if backend is unreachable
  const { data: healthData, isLoading: isHealthLoading, isError: isHealthError } = useQuery({
    queryKey: ['backendHealth'],
    queryFn: () => checkHealth({ timeoutMs: 5000 }),
    refetchInterval: 30000,
    staleTime: 25000,
    retry: 1,
  });

  const isBackendHealthy = !isHealthLoading && !isHealthError && healthData?.status === 'healthy';

  const announce = (message: string) => {
    setAriaAnnouncement(message);
  };

  return (
    <div className={styles.appShell}>
      {/* Skip Navigation Link for Keyboard Accessibility */}
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>

      {/* Screen Reader Live Region */}
      <div
        id="aria-status-announcer"
        role="status"
        aria-live="polite"
        className="sr-only"
      >
        {ariaAnnouncement}
      </div>

      <ScrollAndFocusManager announce={announce} />

      {/* Sticky Header */}
      <header className={styles.header} role="banner">
        <div className={styles.headerInner}>
          <NavLink to="/scan" className={styles.brandLink}>
            <img
              src="/assets/logo.svg"
              alt=""
              width="36"
              height="36"
              className={styles.brandLogo}
              aria-hidden="true"
            />
            <div className={styles.brandTitles}>
              <span className={styles.brandName}>LeafScan V2</span>
              <span className={styles.brandTagline}>Foliar Disease Screening</span>
            </div>
          </NavLink>

          <nav className={styles.nav} aria-label="Main Navigation">
            <NavLink
              to="/scan"
              className={({ isActive }) =>
                `${styles.navLink} ${isActive ? styles.active : ''}`
              }
            >
              Scan
            </NavLink>
            <NavLink
              to="/history"
              className={({ isActive }) =>
                `${styles.navLink} ${isActive ? styles.active : ''}`
              }
            >
              History
            </NavLink>
            <NavLink
              to="/about"
              className={({ isActive }) =>
                `${styles.navLink} ${isActive ? styles.active : ''}`
              }
            >
              About
            </NavLink>
            <NavLink
              to="/system"
              className={({ isActive }) =>
                `${styles.navLink} ${isActive ? styles.active : ''}`
              }
            >
              System
            </NavLink>
          </nav>
        </div>
      </header>

      {/* Main Content Landmark */}
      <main id="main-content" className={styles.main} tabIndex={-1}>
        <Routes>
          <Route path="/" element={<Navigate to="/scan" replace />} />
          <Route
            path="/scan"
            element={
              <ScanRoute
                isBackendHealthy={isBackendHealthy}
                announce={announce}
              />
            }
          />
          <Route path="/result/:id" element={<ResultRoute announce={announce} />} />
          <Route path="/history" element={<HistoryRoute announce={announce} />} />
          <Route path="/about" element={<AboutRoute />} />
          <Route path="/system" element={<SystemRoute />} />
          <Route path="*" element={<Navigate to="/scan" replace />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className={styles.footer} role="contentinfo">
        <div className={styles.footerInner}>
          <div className={styles.footerNotice}>
            <strong>Agricultural Advisory Notice:</strong> LeafScan V2 automated diagnostic
            predictions provide decision-support information only and do not replace professional
            on-site plant pathology diagnosis. Always consult certified agricultural extension
            officers and strictly adhere to locally registered pesticide label instructions.
          </div>
          <div className={styles.footerMeta}>
            <span>Model: MobileNetV2 + CBAM Attention (Opset 17)</span>
            <span>·</span>
            <span>FastAPI Backend (Port 8000)</span>
            <span>·</span>
            <span>React 18 & TypeScript Single Page Architecture</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <HashRouter>
        <AppContent />
      </HashRouter>
    </QueryClientProvider>
  );
};

export default App;
