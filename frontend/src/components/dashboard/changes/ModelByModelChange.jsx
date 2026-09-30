import React from 'react';
import { Layers, ShieldCheck, ArrowUpRight, ArrowDownRight, ArrowRight } from 'lucide-react';

export default function ModelByModelChange({ modelChanges, unit }) {
  const models = modelChanges || [];

  return (
    <div className="overview-card model-by-model-change-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">MULTI-MODEL RUN DELTAS</span>
          <h3 className="overview-card-title">SOURCE MODEL FORECAST ADJUSTMENTS</h3>
        </div>
        <div className="models-icon-pill">
          <Layers size={15} color="#0284c7" />
        </div>
      </div>

      <div className="model-changes-table-wrap">
        <table className="model-changes-table">
          <thead>
            <tr>
              <th className="th-system">Model System</th>
              <th className="th-num text-right">Previous Run ({unit})</th>
              <th className="th-num text-right">Current Run ({unit})</th>
              <th className="th-change text-right">Run Shift (Δ)</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const isBlend = !!m.isBlend;
              const isUp = m.change > 0;
              const isDown = m.change < 0;
              const DirectionIcon = isUp ? ArrowUpRight : isDown ? ArrowDownRight : ArrowRight;

              return (
                <tr key={m.id} className={`model-change-row ${isBlend ? 'is-blend-row' : ''}`}>
                  <td className="td-system">
                    <div className="system-cell-group">
                      {isBlend && <ShieldCheck size={14} className="blend-icon-table" />}
                      <span className={isBlend ? 'blend-name-highlight' : 'model-name-standard'}>
                        {m.name}
                      </span>
                      <span className="model-type-subtag">{m.type}</span>
                    </div>
                  </td>

                  <td className="td-num text-right monospace text-muted">
                    {m.previous.toFixed(1)}
                  </td>

                  <td className="td-num text-right monospace">
                    <strong className={isBlend ? 'highlight-blend-curr' : ''}>{m.current.toFixed(1)}</strong>
                  </td>

                  <td className="td-change text-right">
                    <span className={`delta-tag ${isUp ? 'delta-up' : isDown ? 'delta-down' : 'delta-zero'}`}>
                      <DirectionIcon size={12} strokeWidth={2.5} />
                      <span>{m.change > 0 ? `+${m.change.toFixed(1)}` : m.change.toFixed(1)} {unit}</span>
                    </span>
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
