import React from 'react';
import { ShieldCheck, Cpu, ArrowRight } from 'lucide-react';

export default function SkyCastBlendChange({ weightChanges, activeVarChange }) {
  const weights = weightChanges || [];
  const { previous, current, change, unit, name } = activeVarChange || {
    previous: 36.2,
    current: 42.0,
    change: 5.8,
    unit: 'mm',
    name: 'Rainfall',
  };

  return (
    <div className="overview-card skycast-blend-change-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">BLENDER WEIGHT DYNAMICS</span>
          <h3 className="overview-card-title">SKYCAST BLEND EVOLUTION & WEIGHT REBALANCING</h3>
        </div>
        <div className="blend-pill-badge">
          <ShieldCheck size={16} color="#0284c7" />
          <span>Adaptive Multi-Model Synthesis</span>
        </div>
      </div>

      <div className="blend-change-layout">
        {/* Left: Net Blend Change Metric */}
        <div className="blend-net-shift-box">
          <span className="b-shift-label">Blended {name} Run Delta</span>
          <div className="b-shift-values-row">
            <span className="b-prev-num">{previous.toFixed(1)} {unit}</span>
            <ArrowRight size={18} className="b-shift-arrow" />
            <span className="b-curr-num">{current.toFixed(1)} {unit}</span>
          </div>
          <span className="b-shift-badge">
            Net Change: {change > 0 ? `+${change.toFixed(1)}` : change.toFixed(1)} {unit}
          </span>
        </div>

        {/* Right: Model Weight Distribution Shift */}
        <div className="weights-evolution-column">
          <span className="weights-col-title">Constituent Model Weight Shift ($W_m$)</span>
          <div className="weights-bars-list">
            {weights.map((w) => {
              const isGainer = w.delta > 0;
              const isLoser = w.delta < 0;

              return (
                <div key={w.id} className="weight-shift-item">
                  <div className="w-item-header">
                    <span className="w-name">{w.name}</span>
                    <div className="w-vals-cluster">
                      <span className="w-prev">{w.prevWeight}%</span>
                      <span className="w-arrow">→</span>
                      <strong className="w-curr">{w.currWeight}%</strong>
                      <span className={`w-delta-badge ${isGainer ? 'gain' : isLoser ? 'loss' : 'neutral'}`}>
                        {w.delta > 0 ? `+${w.delta}%` : `${w.delta}%`}
                      </span>
                    </div>
                  </div>

                  <div className="w-dual-track">
                    <div
                      className="w-prev-fill"
                      style={{ width: `${w.prevWeight}%` }}
                      title={`Previous: ${w.prevWeight}%`}
                    ></div>
                    <div
                      className="w-curr-fill"
                      style={{ width: `${w.currWeight}%` }}
                      title={`Current: ${w.currWeight}%`}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
