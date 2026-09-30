import React from 'react';
import { Eye, Clock, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function RunsTable({
  runs = [],
  selectedRunId,
  onSelectRun,
}) {
  if (!runs || runs.length === 0) {
    return (
      <div className="forecast-table-empty">
        <Clock size={28} className="text-secondary" />
        <p>No forecast runs match the selected search or filter criteria.</p>
      </div>
    );
  }

  return (
    <div className="forecast-explorer-card runs-table-card">
      <div className="card-top-header">
        <div>
          <h3 className="card-heading">Available Forecast Runs</h3>
          <p className="card-subheading">Select any execution cycle to inspect constituent model logs and pipeline stages</p>
        </div>
      </div>

      <div className="forecast-table-wrapper">
        <table className="forecast-table runs-operational-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Initialization</th>
              <th>Status</th>
              <th>Models</th>
              <th>Horizon</th>
              <th>Duration</th>
              <th>Updated</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => {
              const isSelected = r.id === selectedRunId;
              const isCompleted = r.status === 'Completed';

              return (
                <tr
                  key={r.id}
                  className={`run-row-selectable ${isSelected ? 'row-selected' : ''}`}
                  onClick={() => onSelectRun(r.id)}
                >
                  <td>
                    <div className="run-id-cell">
                      <span className="font-mono font-semibold">{r.runNumber}</span>
                      {r.isLatest && <span className="run-latest-badge">Latest</span>}
                    </div>
                  </td>
                  <td>
                    <span className="font-mono">{r.initializationIst}</span>
                  </td>
                  <td>
                    <span className={`status-badge-compact ${isCompleted ? 'badge-completed' : 'badge-partial'}`}>
                      <span className={`status-indicator-dot ${isCompleted ? 'dot-green' : 'dot-amber'}`}></span>
                      {r.status}
                    </span>
                  </td>
                  <td>
                    <span className="font-mono font-semibold">{r.modelsRatio}</span>
                  </td>
                  <td>
                    <span className="font-mono text-cyan">{r.horizon}</span>
                  </td>
                  <td>
                    <span className="font-mono">{r.durationSec}s</span>
                  </td>
                  <td>
                    <span className="font-mono text-secondary">{r.lastUpdated}</span>
                  </td>
                  <td>
                    <button
                      type="button"
                      className={`run-inspect-btn ${isSelected ? 'btn-active' : ''}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectRun(r.id);
                      }}
                      title="Inspect Run Execution Details"
                    >
                      <Eye size={13} />
                      <span>{isSelected ? 'Inspecting' : 'Inspect'}</span>
                    </button>
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
