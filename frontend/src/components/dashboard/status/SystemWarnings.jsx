import React from 'react';
import { ShieldCheck, AlertTriangle } from 'lucide-react';

export default function SystemWarnings({ warnings = [] }) {
  const hasWarnings = warnings && warnings.length > 0;

  return (
    <div className="forecast-explorer-card warnings-card">
      <div className="card-top-header">
        <div>
          <h3 className="card-heading">Active Warnings</h3>
          <p className="card-subheading">Operational anomalies, ingestion timeouts, and fallback alarms</p>
        </div>
      </div>

      {hasWarnings ? (
        <div className="active-warnings-list">
          {warnings.map((w, idx) => (
            <div key={idx} className="warning-alert-box">
              <AlertTriangle size={16} className="text-amber-400" />
              <div className="warning-text">
                <span className="warning-title">{w.title}</span>
                <p className="warning-desc">{w.description}</p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="no-warnings-box">
          <div className="no-warn-icon-wrap">
            <ShieldCheck size={28} className="text-emerald-400" />
          </div>
          <div className="no-warn-text">
            <h4>No active system warnings.</h4>
            <p>
              All forecast pipelines, multi-model ingestions, and probabilistic calibration routines are operating nominally without latency degradation.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
