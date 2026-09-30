import React from 'react';
import { Activity, ArrowUpRight, ArrowDownRight, ArrowRight } from 'lucide-react';

export default function ChangeByVariable({ variableChanges, onSelectVariable, selectedVariable }) {
  const vars = variableChanges || [];

  return (
    <div className="overview-card change-by-variable-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">MULTI-VARIABLE SUMMARY</span>
          <h3 className="overview-card-title">CHANGE ACROSS METEOROLOGICAL PARAMETERS</h3>
        </div>
        <div className="var-header-pill">
          <Activity size={15} color="#0284c7" />
        </div>
      </div>

      <div className="variable-changes-grid">
        {vars.map((v) => {
          const isSelected = selectedVariable === v.variable;
          const isUp = v.change > 0;
          const isDown = v.change < 0;
          const DirectionIcon = isUp ? ArrowUpRight : isDown ? ArrowDownRight : ArrowRight;

          return (
            <div
              key={v.variable}
              className={`var-change-card ${isSelected ? 'is-selected-card' : ''}`}
              onClick={() => onSelectVariable && onSelectVariable(v.variable)}
              style={{ cursor: 'pointer' }}
            >
              <div className="vcard-top">
                <span className="vcard-name">{v.name}</span>
                <div className={`vcard-badge ${isUp ? 'badge-up' : isDown ? 'badge-down' : 'badge-neutral'}`}>
                  <DirectionIcon size={12} strokeWidth={2.5} />
                  <span>{v.change > 0 ? `+${v.change.toFixed(1)}` : v.change.toFixed(1)} {v.unit}</span>
                </div>
              </div>

              <div className="vcard-comparison-row">
                <div className="vcard-comp-cell">
                  <span className="v-lbl">Previous</span>
                  <span className="v-num">{v.previous.toFixed(1)} {v.unit}</span>
                </div>
                <div className="vcard-arrow">→</div>
                <div className="vcard-comp-cell">
                  <span className="v-lbl">Current</span>
                  <span className="v-num current-val">{v.current.toFixed(1)} {v.unit}</span>
                </div>
              </div>

              <span className="vcard-status-note">{v.status}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
