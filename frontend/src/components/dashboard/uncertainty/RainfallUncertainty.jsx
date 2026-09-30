import React from 'react';
import { CloudRain, Droplet, Sliders, CheckCircle2 } from 'lucide-react';

export default function RainfallUncertainty({ rainfallSpecific, centralForecast, unit }) {
  if (!rainfallSpecific) return null;

  const { probabilityOfPrecipitation, exceedanceThresholds, zeroInflatedParams } = rainfallSpecific;

  return (
    <div className="overview-card rainfall-uncertainty-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">PRECIPITATION CALIBRATION</span>
          <h3 className="overview-card-title">RAINFALL PROBABILISTIC DYNAMICS</h3>
        </div>
        <div className="rain-icon-pill">
          <CloudRain size={16} color="#0284c7" />
        </div>
      </div>

      <div className="rainfall-prob-grid">
        {/* Probability of Precipitation (PoP) */}
        <div className="rain-stat-card">
          <div className="rsc-header">
            <Droplet size={14} className="rsc-icon" />
            <span className="rsc-lbl">Probability of Precipitation</span>
          </div>
          <div className="rsc-val-row">
            <span className="rsc-main-num">{probabilityOfPrecipitation ?? 78}%</span>
          </div>
          <span className="rsc-sub">Probability of measurable rain (≥ 0.1 mm)</span>
        </div>

        {/* Exceedance Probabilities */}
        <div className="rain-thresholds-container">
          <span className="r-thresh-title">Threshold Exceedance Probabilities</span>
          <div className="r-thresh-list">
            {(exceedanceThresholds || []).map((t) => (
              <div key={t.label} className="r-thresh-row">
                <div className="t-row-left">
                  <span className="t-label">{t.label}</span>
                </div>
                <div className="t-row-right">
                  <div className="t-micro-bar">
                    <div
                      className="t-micro-fill"
                      style={{ width: `${t.prob}%` }}
                    ></div>
                  </div>
                  <span className="t-pct monospace">{t.prob}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Zero-Inflated Gamma EMOS Parameters */}
        <div className="rain-params-card">
          <div className="rsc-header">
            <Sliders size={14} className="rsc-icon" />
            <span className="rsc-lbl">Zero-Inflated Gamma Parameters</span>
          </div>
          <div className="r-params-list">
            <div className="param-item">
              <span className="p-name">Zero-Mass Prob $P(Y=0)$:</span>
              <span className="p-val monospace">{zeroInflatedParams?.zeroProbability || '22%'}</span>
            </div>
            <div className="param-item">
              <span className="p-name">Gamma Shape ($k$):</span>
              <span className="p-val monospace">{zeroInflatedParams?.gammaShapeK || '2.45'}</span>
            </div>
            <div className="param-item">
              <span className="p-name">Gamma Scale ($\theta$):</span>
              <span className="p-val monospace">{zeroInflatedParams?.gammaScaleTheta || '17.1 mm'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
