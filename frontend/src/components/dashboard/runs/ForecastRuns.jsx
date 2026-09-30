import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Clock, RefreshCw, AlertTriangle, Database } from 'lucide-react';
import RunsSummaryCards from './RunsSummaryCards';
import RunsFilterBar from './RunsFilterBar';
import RunsTable from './RunsTable';
import RunDetailPanel from './RunDetailPanel';
import { getForecastRunsData, generateFallbackForecastRuns } from '../../../services/api';

export default function ForecastRuns() {
  const [runsData, setRunsData] = useState(() => generateFallbackForecastRuns());
  const [selectedRunId, setSelectedRunId] = useState('run-20260930-1400');
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  // Client-side filters
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('all');
  const [selectedHorizon, setSelectedHorizon] = useState('all');

  const fetchRuns = useCallback(async () => {
    setIsLoading(true);
    setHasError(false);

    try {
      const res = await getForecastRunsData();
      if (res && res.data && res.data.runs) {
        setRunsData(res.data);
        // Maintain selection or select first
        if (!res.data.runs.some((r) => r.id === selectedRunId)) {
          setSelectedRunId(res.data.runs[0]?.id || null);
        }
      } else {
        setHasError(true);
        setErrorMessage('Unable to load forecast run information.');
      }
    } catch (err) {
      console.error('[ForecastRuns] fetch error:', err);
      setHasError(true);
      setErrorMessage(err.message || 'Unable to load forecast run information.');
    } finally {
      setIsLoading(false);
    }
  }, [selectedRunId]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  // Filtered runs list
  const filteredRuns = useMemo(() => {
    if (!runsData?.runs) return [];

    return runsData.runs.filter((run) => {
      // 1. Search Query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const matchesId = run.runNumber.toLowerCase().includes(q) || run.id.toLowerCase().includes(q);
        const matchesTime = run.initializationIst.toLowerCase().includes(q);
        if (!matchesId && !matchesTime) return false;
      }

      // 2. Status
      if (selectedStatus !== 'all' && run.status !== selectedStatus) {
        return false;
      }

      // 3. Horizon
      if (selectedHorizon !== 'all' && run.horizon !== selectedHorizon) {
        return false;
      }

      return true;
    });
  }, [runsData, searchQuery, selectedStatus, selectedHorizon]);

  // Selected run object
  const activeRun = useMemo(() => {
    if (!runsData?.runs) return null;
    return runsData.runs.find((r) => r.id === selectedRunId) || runsData.runs[0] || null;
  }, [runsData, selectedRunId]);

  const handleResetFilters = () => {
    setSearchQuery('');
    setSelectedStatus('all');
    setSelectedHorizon('all');
  };

  return (
    <div className="forecast-runs-page-container">
      {/* 1. Page Header */}
      <div className="overview-header runs-header">
        <div className="overview-header-left">
          <h1 className="overview-title">Forecast Runs</h1>
          <p className="overview-subtitle">
            Monitor forecast generation, model runs and pipeline execution
          </p>
        </div>

        <div className="overview-header-right">
          <div className="runs-header-telemetry-pill">
            <span className="telemetry-lbl">Latest Run:</span>
            <span className="telemetry-time">
              {runsData?.summary?.initializationIst || '14:00 IST'}
            </span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-status-live">
              <span className="live-status-dot status-green"></span>
              Pipeline Operational
            </span>
          </div>
          <button
            type="button"
            className="runs-refresh-btn"
            onClick={fetchRuns}
            disabled={isLoading}
            title="Refresh forecast runs from backend"
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
            <h4>Unable to load forecast run information.</h4>
            <p>{errorMessage || 'The backend operational diagnostics service is currently unreachable.'}</p>
          </div>
          <button type="button" className="error-retry-btn" onClick={fetchRuns}>
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      ) : !runsData || !runsData.runs || runsData.runs.length === 0 ? (
        /* Empty State */
        <div className="overview-placeholder-container">
          <div className="placeholder-box">
            <Database size={36} color="#64748b" />
            <h3>No forecast runs available.</h3>
            <p>No operational cycles have been recorded in the data provenance ledger.</p>
            <button className="placeholder-back-btn" onClick={fetchRuns}>
              <RefreshCw size={14} />
              <span>Refresh Runs</span>
            </button>
          </div>
        </div>
      ) : (
        /* Operational Content */
        <div className={`runs-content-flow ${isLoading ? 'is-refreshing' : ''}`}>
          {/* Summary Cards */}
          <section className="runs-section">
            <RunsSummaryCards summary={runsData.summary} />
          </section>

          {/* Filter Bar */}
          <section className="runs-section">
            <RunsFilterBar
              searchQuery={searchQuery}
              onSearchChange={setSearchQuery}
              selectedStatus={selectedStatus}
              onStatusChange={setSelectedStatus}
              selectedHorizon={selectedHorizon}
              onHorizonChange={setSelectedHorizon}
              onResetFilters={handleResetFilters}
              totalRunsCount={runsData.runs.length}
              filteredRunsCount={filteredRuns.length}
            />
          </section>

          {/* Runs Table & Selected Run Details */}
          <section className="runs-section runs-layout-grid">
            <div className="runs-table-column">
              <RunsTable
                runs={filteredRuns}
                selectedRunId={selectedRunId}
                onSelectRun={setSelectedRunId}
              />
            </div>
            <div className="runs-detail-column">
              <RunDetailPanel
                run={activeRun}
                onClose={() => {}}
              />
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
