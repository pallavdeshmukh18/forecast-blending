import React from 'react';
import {
  Info,
  Clock,
  CheckCircle2,
  AlertTriangle,
  Cpu,
  Layers,
  ChevronRight,
  ShieldCheck,
  RotateCcw
} from 'lucide-react';
import PipelineProgressTracker from './PipelineProgressTracker';

export default function RunDetailPanel({ run, onClose }) {
  if (!run) {
    return (
      <div className="forecast-explorer-card run-detail-empty">
        <Info size={24} className="text-secondary" />
        <p>Select a forecast run from the table to inspect execution telemetry and model metrics.</p>
      </div>
    );
  }

  const isCompleted = run.status === 'Completed';
  const isPartial = run.status === 'Partial';

  return (
    <div className="forecast-explorer-card run-detail-panel">
      {/* Panel Header */}
      <div className="card-top-header run-detail-header">
        <div className="detail-header-left">
          <div className="run-id-badge-wrap">
            <span className="run-id-title font-mono">{run.runNumber}</span>
            {run.isLatest && <span className="latest-tag-pill">Latest Cycle</span>}
          </div>
          <span className={`kpi-status-badge ${isCompleted ? 'badge-completed' : 'badge-partial'}`}>
            <span className={`status-indicator-dot ${isCompleted ? 'dot-green' : 'dot-amber'}`}></span>
            {run.status}
          </span>
        </div>
        <div className="detail-header-meta">
          <span className="text-secondary">Execution Duration:</span>
          <strong className="text-white font-mono">{run.durationSec}s</strong>
        </div>
      </div>

      {/* 1. Run Metadata Grid */}
      <div className="run-meta-summary-grid">
        <div className="run-meta-cell">
          <span className="meta-cell-label">Initialization Time</span>
          <span className="meta-cell-val font-mono">{run.initializationIst}</span>
          <span className="meta-cell-sub">{new Date(run.initializationTime).toUTCString().slice(0, 22)}</span>
        </div>
        <div className="run-meta-cell">
          <span className="meta-cell-label">Created / Ingested</span>
          <span className="meta-cell-val font-mono">{run.createdTime.split('T')[1].replace('Z', ' UTC')}</span>
          <span className="meta-cell-sub">Pipeline Started</span>
        </div>
        <div className="run-meta-cell">
          <span className="meta-cell-label">Completed Timestamp</span>
          <span className="meta-cell-val font-mono">{run.completionTime.split('T')[1].replace('Z', ' UTC')}</span>
          <span className="meta-cell-sub">Published to Store</span>
        </div>
        <div className="run-meta-cell">
          <span className="meta-cell-label">Max Lead Time</span>
          <span className="meta-cell-val font-mono text-cyan">{run.horizon}</span>
          <span className="meta-cell-sub">Synchronized Grid</span>
        </div>
      </div>

      {/* 2. Pipeline Progress Workflow */}
      <div className="run-detail-sub-section">
        <div className="section-title-row">
          <Layers size={14} className="text-blue-400" />
          <h4 className="sub-section-title">End-to-End Pipeline Execution Stages</h4>
        </div>
        <PipelineProgressTracker stages={run.stages} />
      </div>

      {/* 3. Constituent Models Execution Table */}
      <div className="run-detail-sub-section">
        <div className="section-title-row">
          <Cpu size={14} className="text-purple-400" />
          <h4 className="sub-section-title">Constituent NWP & AI Models Execution ({run.modelsProcessedCount} / {run.totalModelsCount})</h4>
        </div>
        <div className="models-execution-table-wrap">
          <table className="forecast-table models-run-table">
            <thead>
              <tr>
                <th>Model System</th>
                <th>Architecture</th>
                <th>Status</th>
                <th>Processing Time</th>
                <th>Network Latency</th>
              </tr>
            </thead>
            <tbody>
              {run.models.map((m) => {
                const modelOk = m.status === 'Completed';
                const modelDelayed = m.status === 'Delayed';
                return (
                  <tr key={m.id}>
                    <td>
                      <span className="model-name font-semibold">{m.name}</span>
                    </td>
                    <td>
                      <span className="model-type-tag">{m.type}</span>
                    </td>
                    <td>
                      <span className={`status-badge-compact ${modelOk ? 'badge-completed' : 'badge-partial'}`}>
                        <span className={`status-indicator-dot ${modelOk ? 'dot-green' : 'dot-amber'}`}></span>
                        {m.status}
                      </span>
                    </td>
                    <td>
                      <span className="font-mono">{m.durationSec != null ? `${m.durationSec}s` : '—'}</span>
                    </td>
                    <td>
                      <span className="font-mono text-secondary">{m.latency}</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Run Issues & Warnings (if present) */}
      {run.issues ? (
        <div className="run-issue-callout">
          <div className="issue-icon-wrap">
            <AlertTriangle size={18} className="text-amber-400" />
          </div>
          <div className="issue-content-block">
            <div className="issue-header-row">
              <span className="issue-title">Pipeline Notice: {run.issues.severity}</span>
              <span className="issue-timestamp font-mono text-secondary">
                {run.issues.timestamp.split('T')[1].replace('Z', ' UTC')}
              </span>
            </div>
            <p className="issue-msg">{run.issues.message}</p>
            <div className="issue-details-grid">
              <div>
                <span className="issue-lbl">Failed / Delayed System:</span>
                <span className="issue-val font-semibold">{run.issues.failedModel}</span>
              </div>
              <div>
                <span className="issue-lbl">Affected Pipeline Stage:</span>
                <span className="issue-val">{run.issues.failedStage}</span>
              </div>
            </div>
            {run.issues.retryStatus && (
              <div className="issue-recovery-pill">
                <RotateCcw size={12} className="text-emerald-400" />
                <span>{run.issues.retryStatus}</span>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="run-healthy-notice">
          <ShieldCheck size={16} className="text-emerald-400" />
          <span>All 5 forecasting systems completed nominal execution with zero anomalies or fallback events.</span>
        </div>
      )}
    </div>
  );
}
