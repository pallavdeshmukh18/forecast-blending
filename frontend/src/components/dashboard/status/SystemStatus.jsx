import React, { useState, useEffect, useCallback } from 'react';
import { Server, RefreshCw, AlertTriangle, Database } from 'lucide-react';
import OverallStatusCard from './OverallStatusCard';
import ServiceStatusGrid from './ServiceStatusGrid';
import ModelAvailabilityTable from './ModelAvailabilityTable';
import SystemDiagnosticsCard from './SystemDiagnosticsCard';
import SystemWarnings from './SystemWarnings';
import RecentEventsLog from './RecentEventsLog';
import { getSystemStatusData, generateFallbackSystemStatus } from '../../../services/api';

export default function SystemStatus() {
  const [statusData, setStatusData] = useState(() => generateFallbackSystemStatus());
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const fetchStatus = useCallback(async () => {
    setIsLoading(true);
    setHasError(false);

    try {
      const res = await getSystemStatusData();
      if (res && res.data && res.data.overall) {
        setStatusData(res.data);
      } else {
        setHasError(true);
        setErrorMessage('Unable to retrieve system status.');
      }
    } catch (err) {
      console.error('[SystemStatus] fetch error:', err);
      setHasError(true);
      setErrorMessage(err.message || 'Unable to retrieve system status.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  return (
    <div className="system-status-page-container">
      {/* 1. Header */}
      <div className="overview-header status-header">
        <div className="overview-header-left">
          <h1 className="overview-title">System Status</h1>
          <p className="overview-subtitle">
            Monitor the health of BharatWeather services and forecasting infrastructure
          </p>
        </div>

        <div className="overview-header-right">
          <div className="status-header-telemetry-pill">
            <span className="telemetry-lbl">Last Check:</span>
            <span className="telemetry-time">
              {statusData?.overall?.lastHealthCheck || '14:02 IST'}
            </span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-status-live">
              <span className="live-status-dot status-green"></span>
              All Systems Monitored
            </span>
          </div>
          <button
            type="button"
            className="runs-refresh-btn"
            onClick={fetchStatus}
            disabled={isLoading}
            title="Probe all system health endpoints"
          >
            <RefreshCw size={14} className={isLoading ? 'spinner-icon' : ''} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* 2. Error State */}
      {hasError ? (
        <div className="forecast-error-box">
          <AlertTriangle size={28} className="error-icon" />
          <div className="error-text-content">
            <h4>Unable to retrieve system status.</h4>
            <p>{errorMessage || 'The backend health probe service could not be reached.'}</p>
          </div>
          <button type="button" className="error-retry-btn" onClick={fetchStatus}>
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      ) : !statusData ? (
        <div className="overview-placeholder-container">
          <div className="placeholder-box">
            <Database size={36} color="#64748b" />
            <h3>Unable to retrieve system status.</h3>
            <p>System diagnostic telemetry is currently unavailable.</p>
            <button className="placeholder-back-btn" onClick={fetchStatus}>
              <RefreshCw size={14} />
              <span>Retry Probe</span>
            </button>
          </div>
        </div>
      ) : (
        /* Operational Content */
        <div className={`status-content-flow ${isLoading ? 'is-refreshing' : ''}`}>
          {/* Section 2: Overall System Status */}
          <section className="status-section">
            <OverallStatusCard overall={statusData.overall} />
          </section>

          {/* Section 3: Core Service Status Grid */}
          <section className="status-section">
            <ServiceStatusGrid services={statusData.services} />
          </section>

          {/* Section 4: Forecast Model Availability */}
          <section className="status-section">
            <ModelAvailabilityTable models={statusData.models} />
          </section>

          {/* Section 6, 7 & 9: Diagnostics, Data Freshness & Active Warnings in 2-Column Grid */}
          <section className="status-section two-column-grid">
            <SystemDiagnosticsCard
              diagnostics={statusData.diagnostics}
              apiEndpoints={statusData.apiEndpoints}
            />
            <SystemWarnings warnings={statusData.warnings} />
          </section>

          {/* Section 8: Recent System Events */}
          <section className="status-section">
            <RecentEventsLog events={statusData.recentEvents} />
          </section>
        </div>
      )}
    </div>
  );
}
