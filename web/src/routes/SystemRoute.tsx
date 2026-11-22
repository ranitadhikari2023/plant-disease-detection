/**
 * web/src/routes/SystemRoute.tsx
 * Backend Architecture Visibility and Telemetry Route
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { checkHealth } from '../api/health';
import { API_BASE_URL } from '../api/client';
import type { HealthResponse } from '../api/schema';
import styles from './SystemRoute.module.css';

interface OpenAPIOperation {
  summary?: string;
  description?: string;
  operationId?: string;
}

interface OpenAPISpec {
  openapi: string;
  info: {
    title: string;
    version: string;
    description?: string;
  };
  paths: Record<string, Record<string, OpenAPIOperation>>;
}

export const SystemRoute: React.FC = () => {
  const { data: healthData, isLoading: healthLoading, isError: healthError } = useQuery<HealthResponse>({
    queryKey: ['systemHealth'],
    queryFn: () => checkHealth({ timeoutMs: 5000 }),
    refetchInterval: 30000,
  });

  const { data: openapiData, isLoading: openapiLoading, isError: openapiError } = useQuery<OpenAPISpec>({
    queryKey: ['openapiSpec'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE_URL}/openapi.json`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    },
    staleTime: 60000,
  });

  const isHealthy = !healthLoading && !healthError && healthData?.status === 'healthy';

  // Parse inventory from openapi.json
  const inventory: { method: string; path: string; summary: string; description: string }[] = [];
  if (openapiData?.paths) {
    Object.entries(openapiData.paths).forEach(([pathName, methods]) => {
      Object.entries(methods).forEach(([methodName, op]) => {
        inventory.push({
          method: methodName.toUpperCase(),
          path: pathName,
          summary: op.summary || 'Endpoint',
          description: op.description || '',
        });
      });
    });
  }

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>Backend System & API Architecture</h1>
        <p className={styles.subtitle}>
          Operational telemetry, REST API inventory, and OpenAPI specification from the frozen FastAPI backend.
        </p>
      </header>

      <div className={styles.grid}>
        {/* Card 1: Backend Health Status */}
        <section className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.badge}>Live Telemetry</span>
            <h2 className={styles.cardTitle}>FastAPI Service Health</h2>
          </div>

          <div className={styles.statusRow}>
            <span
              className={`${styles.statusDot} ${isHealthy ? styles.dotOnline : styles.dotOffline}`}
              aria-hidden="true"
            />
            <span className={styles.statusLabel}>
              {healthLoading
                ? 'Checking backend...'
                : isHealthy
                ? 'Backend Operational (Healthy)'
                : 'Service Unavailable'}
            </span>
          </div>

          <div className={styles.table}>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>Reported Status</span>
              <span className={styles.tableVal}>{healthData?.status || 'Unknown'}</span>
            </div>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>Model Version</span>
              <span className={styles.tableVal}>{healthData?.model_version || 'v2'}</span>
            </div>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>Backbone Architecture</span>
              <span className={styles.tableVal}>
                {healthData?.architecture || 'MobileNetV2 + CBAM Attention'}
              </span>
            </div>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>Models Loaded (In-Memory)</span>
              <span className={styles.tableVal}>
                {healthData?.models_loaded === true
                  ? 'Yes (Cached Singleton)'
                  : 'Lazy (Loads on first inference)'}
              </span>
            </div>
          </div>
        </section>

        {/* Card 2: Environment & Networking Configuration */}
        <section className={styles.card}>
          <div className={styles.cardHeader}>
            <span className={styles.badge}>Networking</span>
            <h2 className={styles.cardTitle}>Frontend Connection Configuration</h2>
          </div>

          <div className={styles.table}>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>VITE_API_BASE_URL</span>
              <span className={styles.tableVal}>
                <code>{API_BASE_URL || '(Same-Origin / Proxy)'}</code>
              </span>
            </div>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>Dev Proxy Target</span>
              <span className={styles.tableVal}>
                <code>http://127.0.0.1:8000</code>
              </span>
            </div>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>Inference Timeout</span>
              <span className={styles.tableVal}>30 seconds</span>
            </div>
            <div className={styles.tableRow}>
              <span className={styles.tableKey}>Health Check Polling</span>
              <span className={styles.tableVal}>Every 30s (5s timeout)</span>
            </div>
          </div>

          <div className={styles.docsSection}>
            <h3 className={styles.docsTitle}>Interactive Backend Documentation</h3>
            <div className={styles.docsLinks}>
              <a
                href={`${API_BASE_URL || ''}/docs`}
                target="_blank"
                rel="noreferrer"
                className="btn btn-secondary btn-sm"
              >
                Swagger UI (/docs) ↗
              </a>
              <a
                href={`${API_BASE_URL || ''}/redoc`}
                target="_blank"
                rel="noreferrer"
                className="btn btn-secondary btn-sm"
              >
                ReDoc UI (/redoc) ↗
              </a>
              <a
                href={`${API_BASE_URL || ''}/openapi.json`}
                target="_blank"
                rel="noreferrer"
                className="btn btn-secondary btn-sm"
              >
                Raw OpenAPI JSON ↗
              </a>
            </div>
          </div>
        </section>
      </div>

      {/* Card 3: API Endpoint Inventory */}
      <section className={styles.inventoryCard}>
        <div className={styles.cardHeader}>
          <span className={styles.badge}>Endpoint Inventory</span>
          <h2 className={styles.cardTitle}>Discovered API Endpoints (/openapi.json)</h2>
        </div>

        {openapiLoading ? (
          <div className={styles.loadingText}>Fetching OpenAPI specification from backend...</div>
        ) : openapiError || inventory.length === 0 ? (
          <div className={styles.inventoryFallback}>
            <p>Could not fetch live OpenAPI JSON from backend server. Known verified routes:</p>
            <ul className={styles.fallbackList}>
              <li>
                <code>GET /health</code> — Lightweight system health telemetry
              </li>
              <li>
                <code>POST /v2/diagnose</code> — Foliar specimen classification with CBAM attention
              </li>
              <li>
                <code>GET /</code> — Serves modernized Single Page Application
              </li>
            </ul>
          </div>
        ) : (
          <div className={styles.endpointList}>
            {inventory.map((ep, idx) => (
              <div key={idx} className={styles.endpointRow}>
                <div className={styles.epHeader}>
                  <span
                    className={`${styles.methodBadge} ${
                      ep.method === 'POST' ? styles.postBadge : styles.getBadge
                    }`}
                  >
                    {ep.method}
                  </span>
                  <code className={styles.epPath}>{ep.path}</code>
                  <span className={styles.epSummary}>{ep.summary}</span>
                </div>
                {ep.description && <p className={styles.epDesc}>{ep.description}</p>}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
