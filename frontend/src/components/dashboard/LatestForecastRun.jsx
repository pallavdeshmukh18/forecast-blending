import React from 'react';
import { CheckCircle2, Clock, Cpu, ArrowRight } from 'lucide-react';

export default function LatestForecastRun({ runData, onNavigateToRuns }) {
  const runId = runData?.runId || '#20260930-1400';
  const initTime = runData?.initTime || '14:00 IST';
  const status = runData?.status || 'Complete';
  const sources = runData?.sources || '5 / 5';
  const blendingStatus = runData?.blending || 'Complete';
  const calibrationStatus = runData?.calibration || 'Complete';
  const duration = runData?.duration || '42 sec';

  return (
    <div className="overview-card latest-run-card-full">
      <div className="overview-card-header">
        <div className="latest-run-title-wrap">
          <Clock size={16} className="clock-icon" />
          <h3 className="overview-card-title">LATEST FORECAST RUN</h3>
        </div>

        <button
          type="button"
          className="run-action-btn"
          onClick={onNavigateToRuns}
          title="Open Forecast Runs Audit"
        >
          <span>View run</span>
          <ArrowRight size={14} />
        </button>
      </div>

      <div className="run-metrics-grid">
        <div className="run-stat-cell">
          <span className="run-stat-label">Run ID</span>
          <span className="run-stat-value mono">{runId}</span>
        </div>

        <div className="run-stat-cell">
          <span className="run-stat-label">Initialization</span>
          <span className="run-stat-value">{initTime}</span>
        </div>

        <div className="run-stat-cell">
          <span className="run-stat-label">Status</span>
          <span className="run-stat-status-badge">
            <span className="run-status-dot"></span>
            <span>{status}</span>
          </span>
        </div>

        <div className="run-stat-cell">
          <span className="run-stat-label">Sources processed</span>
          <span className="run-stat-value">{sources}</span>
        </div>

        <div className="run-stat-cell">
          <span className="run-stat-label">Blending</span>
          <span className="run-stat-val-inline">
            <CheckCircle2 size={13} color="#059669" />
            <span>{blendingStatus}</span>
          </span>
        </div>

        <div className="run-stat-cell">
          <span className="run-stat-label">Calibration</span>
          <span className="run-stat-val-inline">
            <CheckCircle2 size={13} color="#059669" />
            <span>{calibrationStatus}</span>
          </span>
        </div>

        <div className="run-stat-cell">
          <span className="run-stat-label">Pipeline duration</span>
          <span className="run-stat-value highlight">{duration}</span>
        </div>
      </div>
    </div>
  );
}
