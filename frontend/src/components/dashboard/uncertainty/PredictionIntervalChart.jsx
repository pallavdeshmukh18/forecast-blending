import React, { useState } from 'react';
import { TrendingUp, Clock, Info } from 'lucide-react';

export default function PredictionIntervalChart({ horizonsList, unit, varName }) {
  const points = horizonsList || [];
  const [activePointIndex, setActivePointIndex] = useState(2); // Default to +24h

  if (points.length === 0) return null;

  // Chart coordinate geometry
  const width = 640;
  const height = 240;
  const padLeft = 46;
  const padRight = 30;
  const padTop = 30;
  const padBottom = 40;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  // Calculate global min and max across all points for bounds
  const allValues = [];
  points.forEach((p) => {
    allValues.push(p.forecast, p.low80, p.high80);
  });
  const minVal = Math.max(0, Math.floor(Math.min(...allValues) * 0.85));
  const maxVal = Math.ceil(Math.max(...allValues) * 1.15) || 10;
  const valRange = maxVal - minVal || 1;

  const getX = (idx) => padLeft + (idx / (points.length - 1)) * chartW;
  const getY = (val) => padTop + chartH - ((val - minVal) / valRange) * chartH;

  // Construct SVG polygon path for shaded 80% interval envelope
  const upperPath = points.map((p, i) => `${getX(i)},${getY(p.high80)}`).join(' L ');
  const lowerPathRev = points
    .slice()
    .reverse()
    .map((p, i) => `${getX(points.length - 1 - i)},${getY(p.low80)}`)
    .join(' L ');
  const envelopeAreaD = `M ${upperPath} L ${lowerPathRev} Z`;

  // Construct central forecast path
  const linePathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(i)},${getY(p.forecast)}`).join(' ');

  const activePoint = points[activePointIndex] || points[0];

  return (
    <div className="overview-card prediction-interval-chart-card">
      <div className="overview-card-header">
        <div>
          <span className="card-section-tag">TIME SERIES CALIBRATION</span>
          <h3 className="overview-card-title">PREDICTION INTERVAL & UNCERTAINTY BAND</h3>
        </div>
        <div className="chart-legend-row">
          <div className="legend-item">
            <span className="legend-line forecast-line"></span>
            <span>Central Blend</span>
          </div>
          <div className="legend-item">
            <span className="legend-swatch interval-swatch"></span>
            <span>80% Prediction Band</span>
          </div>
        </div>
      </div>

      <div className="chart-svg-container">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="uncertainty-chart-svg"
        >
          {/* Horizontal grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
            const yVal = minVal + ratio * valRange;
            const yPos = getY(yVal);
            return (
              <g key={ratio} className="chart-grid-line-group">
                <line
                  x1={padLeft}
                  y1={yPos}
                  x2={width - padRight}
                  y2={yPos}
                  stroke="#e2e8f0"
                  strokeDasharray="4 4"
                  strokeWidth={1}
                />
                <text
                  x={padLeft - 8}
                  y={yPos + 4}
                  textAnchor="end"
                  fontSize="10"
                  fill="#94a3b8"
                  fontFamily="monospace"
                >
                  {Math.round(yVal)}
                </text>
              </g>
            );
          })}

          {/* Shaded 80% Prediction Interval Envelope */}
          <path
            d={envelopeAreaD}
            fill="rgba(56, 189, 248, 0.22)"
            stroke="rgba(2, 132, 199, 0.45)"
            strokeWidth={1.5}
            className="chart-envelope-path"
          />

          {/* Central Blended Forecast Line */}
          <path
            d={linePathD}
            fill="none"
            stroke="#0284c7"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="chart-central-line"
          />

          {/* Interactive Data Points */}
          {points.map((p, i) => {
            const cx = getX(i);
            const cy = getY(p.forecast);
            const isSelected = i === activePointIndex;

            return (
              <g
                key={p.horizon}
                className="chart-point-group"
                onClick={() => setActivePointIndex(i)}
                style={{ cursor: 'pointer' }}
              >
                {/* Vertical guide line on active point */}
                {isSelected && (
                  <line
                    x1={cx}
                    y1={padTop}
                    x2={cx}
                    y2={padTop + chartH}
                    stroke="#0284c7"
                    strokeWidth={1.5}
                    strokeDasharray="2 2"
                  />
                )}

                {/* Point circle */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 6 : 4}
                  fill={isSelected ? '#0284c7' : '#ffffff'}
                  stroke="#0284c7"
                  strokeWidth={isSelected ? 3 : 2}
                  className="chart-point-dot"
                />

                {/* Horizon label on bottom axis */}
                <text
                  x={cx}
                  y={height - padBottom + 20}
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight={isSelected ? '800' : '600'}
                  fill={isSelected ? '#0f172a' : '#64748b'}
                >
                  {p.horizon}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Selected Horizon Telemetry Card */}
        {activePoint && (
          <div className="chart-telemetry-badge">
            <div className="badge-horizon-pill">
              <Clock size={12} />
              <span>{activePoint.horizon}</span>
            </div>
            <div className="badge-stat">
              <span className="b-lbl">Forecast:</span>
              <span className="b-val">{activePoint.forecast} {unit}</span>
            </div>
            <div className="badge-stat">
              <span className="b-lbl">80% Interval:</span>
              <span className="b-val">[{activePoint.low80} — {activePoint.high80} {unit}]</span>
            </div>
            <div className="badge-stat">
              <span className="b-lbl">Spread:</span>
              <span className="b-val">{activePoint.spread} {unit}</span>
            </div>
          </div>
        )}
      </div>

      <div className="chart-footer-note">
        <Info size={13} className="info-icon" />
        <span>
          Shaded envelope represents the 80% Bayesian predictive interval. Wider bands indicate greater uncertainty in physical parameterizations at longer lead times.
        </span>
      </div>
    </div>
  );
}
