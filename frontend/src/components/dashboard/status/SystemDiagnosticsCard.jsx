import React from 'react';
import { Activity, Clock, Database, CheckCircle2, ShieldCheck } from 'lucide-react';

export default function SystemDiagnosticsCard({ diagnostics, apiEndpoints = [] }) {
  if (!diagnostics) return null;

  return (
    <div className="forecast-explorer-card diagnostics-card">
      <div className="card-top-header">
        <div>
          <h3 className="card-heading">Pipeline Diagnostics & Data Freshness</h3>
          <p className="card-subheading">Data ingestion state, synchronization freshness, and API endpoint probes</p>
        </div>
      </div>

      {/* Diagnostics KPI Grid */}
      <div className="diagnostics-metrics-grid">
        <div className="diag-metric-item">
          <span className="diag-lbl">Data Freshness</span>
          <span className="diag-val text-emerald-400 font-mono">{diagnostics.dataFreshness}</span>
          <span className="diag-sub">Latest Synoptic Ingestion</span>
        </div>
        <div className="diag-metric-item">
          <span className="diag-lbl">Latest Forecast Run</span>
          <span className="diag-val font-mono">{diagnostics.latestRun}</span>
          <span className="diag-sub">Execution: {diagnostics.processingDurationSec}s</span>
        </div>
        <div className="diag-metric-item">
          <span className="diag-lbl">Pipeline Engine State</span>
          <span className="diag-val font-mono text-cyan">{diagnostics.pipelineState.split('(')[0].trim()}</span>
          <span className="diag-sub">{diagnostics.pipelineState.includes('(') ? diagnostics.pipelineState.match(/\((.*?)\)/)?.[1] : 'Operational'}</span>
        </div>
        <div className="diag-metric-item">
          <span className="diag-lbl">Data Mode</span>
          <span className="diag-val font-mono">{diagnostics.dataMode}</span>
          <span className="diag-sub">{diagnostics.totalProvenanceDatasets} Datasets Tracked</span>
        </div>
      </div>

      {/* API Probes Sub-section */}
      <div className="api-probes-section">
        <div className="section-title-row">
          <Activity size={14} className="text-blue-400" />
          <h4 className="sub-section-title">Backend API Health Probes</h4>
        </div>
        <div className="api-probes-grid">
          {apiEndpoints.map((ep) => (
            <div key={ep.path} className="api-probe-pill">
              <span className="api-probe-path font-mono">{ep.path}</span>
              <span className="api-probe-status">
                <CheckCircle2 size={12} className="text-emerald-400" />
                <span className="text-green font-mono">{ep.responseTime}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
