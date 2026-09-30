import React from 'react';
import { PlayCircle, CheckCircle2, Cpu, Clock, Calendar } from 'lucide-react';

export default function RunsSummaryCards({ summary }) {
  if (!summary) return null;

  const isCompleted = summary.status === 'Completed';

  return (
    <div className="overview-kpis-grid runs-summary-grid">
      {/* 1. Latest Run */}
      <div className="overview-kpi-card run-kpi-card">
        <div className="kpi-header">
          <span className="kpi-label">LATEST RUN</span>
          <div className="kpi-icon-pill icon-blue">
            <PlayCircle size={16} />
          </div>
        </div>
        <div className="kpi-value-row">
          <span className="kpi-main-val font-mono">{summary.latestRunId}</span>
        </div>
        <div className="kpi-sub-row">
          <Calendar size={12} />
          <span>Init: {summary.initializationIst}</span>
          <span className="telemetry-sep">•</span>
          <span>{summary.durationSec}s cycle</span>
        </div>
      </div>

      {/* 2. Run Status */}
      <div className="overview-kpi-card run-kpi-card">
        <div className="kpi-header">
          <span className="kpi-label">RUN STATUS</span>
          <div className={`kpi-icon-pill ${isCompleted ? 'icon-green' : 'icon-amber'}`}>
            <CheckCircle2 size={16} />
          </div>
        </div>
        <div className="kpi-value-row">
          <span className={`kpi-status-badge ${isCompleted ? 'badge-completed' : 'badge-partial'}`}>
            <span className={`status-indicator-dot ${isCompleted ? 'dot-green' : 'dot-amber'}`}></span>
            {summary.status}
          </span>
        </div>
        <div className="kpi-sub-row">
          <span>Last sync at {summary.lastUpdated}</span>
        </div>
      </div>

      {/* 3. Models Processed */}
      <div className="overview-kpi-card run-kpi-card">
        <div className="kpi-header">
          <span className="kpi-label">MODELS PROCESSED</span>
          <div className="kpi-icon-pill icon-purple">
            <Cpu size={16} />
          </div>
        </div>
        <div className="kpi-value-row">
          <span className="kpi-main-val font-mono">{summary.modelsRatio}</span>
          <span className="kpi-unit">active</span>
        </div>
        <div className="kpi-sub-row">
          <span className="text-secondary">ECMWF, GFS, GraphCast, Pangu, WRF</span>
        </div>
      </div>

      {/* 4. Latest Forecast Horizon */}
      <div className="overview-kpi-card run-kpi-card">
        <div className="kpi-header">
          <span className="kpi-label">LATEST FORECAST HORIZON</span>
          <div className="kpi-icon-pill icon-teal">
            <Clock size={16} />
          </div>
        </div>
        <div className="kpi-value-row">
          <span className="kpi-main-val font-mono text-cyan">{summary.horizon}</span>
          <span className="kpi-unit">lead time</span>
        </div>
        <div className="kpi-sub-row">
          <span>Continuous multi-horizon coverage</span>
        </div>
      </div>
    </div>
  );
}
