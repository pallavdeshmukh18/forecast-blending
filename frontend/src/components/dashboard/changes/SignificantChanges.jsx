import React from 'react';
import { AlertTriangle, Tag, CheckCircle2 } from 'lucide-react';

export default function SignificantChanges({ significantChanges }) {
  const items = significantChanges || [];

  return (
    <div className="overview-card significant-changes-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">THRESHOLD ANOMALIES</span>
          <h3 className="overview-card-title">SIGNIFICANT OPERATIONAL RUN CHANGES</h3>
        </div>
        <div className="sig-icon-pill">
          <AlertTriangle size={15} color="#ea580c" />
        </div>
      </div>

      <div className="significant-items-list">
        {items.length > 0 ? (
          items.map((item, idx) => (
            <div key={idx} className="significant-item-row">
              <div className="sig-left">
                <span className="sig-variable-tag">{item.variable}</span>
                <span className="sig-horizon-tag">{item.horizon}</span>
                <span className="sig-model-name">{item.model}</span>
              </div>

              <div className="sig-right">
                <span className="sig-magnitude-num">{item.magnitude}</span>
                <span className="sig-category-pill">{item.tag}</span>
              </div>
            </div>
          ))
        ) : (
          <div className="no-sig-changes">
            <CheckCircle2 size={16} color="#059669" />
            <span>All parameter adjustments remain within nominal tolerance limits (&lt; 5 mm rain, &lt; 1.0°C temp).</span>
          </div>
        )}
      </div>
    </div>
  );
}
