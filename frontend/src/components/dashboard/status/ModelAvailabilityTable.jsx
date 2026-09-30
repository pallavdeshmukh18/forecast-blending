import React from 'react';
import { Cpu, CheckCircle2, Clock, Activity } from 'lucide-react';

export default function ModelAvailabilityTable({ models = [] }) {
  if (!models || models.length === 0) return null;

  return (
    <div className="forecast-explorer-card model-availability-card">
      <div className="card-top-header">
        <div>
          <h3 className="card-heading">Forecast Model Availability</h3>
          <p className="card-subheading">Source NWP & AI weather model health, grid resolution, and refresh intervals</p>
        </div>
      </div>

      <div className="forecast-table-wrapper">
        <table className="forecast-table models-availability-table">
          <thead>
            <tr>
              <th>Model System</th>
              <th>Status</th>
              <th>Architecture</th>
              <th>Spatial Resolution</th>
              <th>Update Frequency</th>
              <th>Response Latency</th>
              <th>Last Run Cycle</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const isOnline = m.status === 'Online';
              const isActive = m.status === 'Active';

              return (
                <tr key={m.id} className={isActive ? 'row-blend-highlight' : ''}>
                  <td>
                    <div className="model-name-cell">
                      <span className="font-semibold text-white">{m.name}</span>
                      {isActive && <span className="blend-indicator-pill">Operational Blend</span>}
                    </div>
                  </td>
                  <td>
                    <span className={`status-badge-compact ${isOnline || isActive ? 'badge-completed' : 'badge-partial'}`}>
                      <span className={`status-indicator-dot ${isOnline ? 'dot-green' : isActive ? 'dot-blue' : 'dot-amber'}`}></span>
                      {m.status}
                    </span>
                  </td>
                  <td>
                    <span className="model-type-tag">{m.type}</span>
                  </td>
                  <td>
                    <span className="font-mono text-secondary">{m.resolution}</span>
                  </td>
                  <td>
                    <span className="font-mono">{m.updateFrequency}</span>
                  </td>
                  <td>
                    <span className="font-mono text-emerald-400">{m.latency}</span>
                  </td>
                  <td>
                    <span className="font-mono text-secondary">{m.lastRun}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
