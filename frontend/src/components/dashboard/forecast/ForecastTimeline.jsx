import React, { useState } from 'react';
import { TrendingUp, Clock } from 'lucide-react';

export default function ForecastTimeline({ forecastData, selectedLeadTime, onLeadTimeChange }) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const timeline = forecastData?.timeline || [
    { horizon: 'Now', hours: 0, value: 8 },
    { horizon: '+6h', hours: 6, value: 12 },
    { horizon: '+12h', hours: 12, value: 18 },
    { horizon: '+24h', hours: 24, value: 42 },
    { horizon: '+48h', hours: 48, value: 31 },
    { horizon: '+72h', hours: 72, value: 15 },
  ];

  const unit = forecastData?.unit ?? 'mm';
  const varLabel = forecastData?.variableLabel ?? 'Rainfall';

  // SVG Chart Geometry
  const width = 640;
  const height = 180;
  const padding = { top: 25, right: 35, bottom: 35, left: 45 };

  const values = timeline.map((t) => t.value);
  const minVal = Math.min(...values, 0);
  const maxVal = Math.max(...values, 10);
  const valRange = Math.max(1, maxVal - minVal);

  const getX = (index) => {
    return padding.left + (index / (timeline.length - 1)) * (width - padding.left - padding.right);
  };

  const getY = (val) => {
    return height - padding.bottom - ((val - minVal) / valRange) * (height - padding.top - padding.bottom);
  };

  // Build SVG Path
  const points = timeline.map((t, idx) => ({
    x: getX(idx),
    y: getY(t.value),
    item: t,
    index: idx,
  }));

  const linePath = points.reduce((acc, p, idx) => {
    if (idx === 0) return `M ${p.x} ${p.y}`;
    // Smooth cubic bezier curve
    const prev = points[idx - 1];
    const cp1x = prev.x + (p.x - prev.x) / 2;
    const cp1y = prev.y;
    const cp2x = prev.x + (p.x - prev.x) / 2;
    const cp2y = p.y;
    return `${acc} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${p.x} ${p.y}`;
  }, '');

  const areaPath = `${linePath} L ${points[points.length - 1].x} ${height - padding.bottom} L ${points[0].x} ${height - padding.bottom} Z`;

  const selectedHours = parseInt(String(selectedLeadTime).replace('+', '').replace('h', ''), 10) || 24;

  return (
    <div className="overview-card forecast-timeline-card-full">
      <div className="overview-card-header">
        <div className="timeline-header-title">
          <TrendingUp size={16} className="timeline-icon" />
          <h3 className="overview-card-title">FORECAST TIMELINE</h3>
          <span className="timeline-variable-badge">{varLabel} ({unit})</span>
        </div>
        <span className="timeline-subtitle-hint">Click a horizon point to inspect that lead time</span>
      </div>

      {/* SVG Chart Area */}
      <div className="timeline-chart-wrapper">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="timeline-svg-chart"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <linearGradient id="timelineGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#0284c7" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          <line
            x1={padding.left}
            y1={padding.top}
            x2={width - padding.right}
            y2={padding.top}
            stroke="rgba(255, 255, 255, 0.08)"
            strokeDasharray="3 3"
          />
          <line
            x1={padding.left}
            y1={(padding.top + height - padding.bottom) / 2}
            x2={width - padding.right}
            y2={(padding.top + height - padding.bottom) / 2}
            stroke="rgba(255, 255, 255, 0.08)"
            strokeDasharray="3 3"
          />
          <line
            x1={padding.left}
            y1={height - padding.bottom}
            x2={width - padding.right}
            y2={height - padding.bottom}
            stroke="rgba(255, 255, 255, 0.15)"
          />

          {/* Area Fill */}
          <path d={areaPath} fill="url(#timelineGradient)" />

          {/* Line Path */}
          <path d={linePath} fill="none" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" />

          {/* Data Points */}
          {points.map((p) => {
            const isSelected = p.item.hours === selectedHours;
            const isHovered = hoveredPoint === p.index;

            return (
              <g
                key={p.item.horizon}
                className="timeline-point-group"
                onClick={() => onLeadTimeChange && onLeadTimeChange(String(p.item.hours))}
                onMouseEnter={() => setHoveredPoint(p.index)}
                onMouseLeave={() => setHoveredPoint(null)}
                style={{ cursor: 'pointer' }}
              >
                {/* Vertical helper line for active/hovered */}
                {(isSelected || isHovered) && (
                  <line
                    x1={p.x}
                    y1={padding.top}
                    x2={p.x}
                    y2={height - padding.bottom}
                    stroke={isSelected ? '#38bdf8' : 'rgba(255,255,255,0.3)'}
                    strokeWidth="1.2"
                    strokeDasharray="2 2"
                  />
                )}

                {/* Point circle */}
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={isSelected ? 6 : 4.5}
                  fill={isSelected ? '#0284c7' : '#0f172a'}
                  stroke={isSelected ? '#38bdf8' : '#38bdf8'}
                  strokeWidth={isSelected ? 3 : 2}
                />

                {/* Point value label */}
                <text
                  x={p.x}
                  y={p.y - 10}
                  textAnchor="middle"
                  fill={isSelected ? '#38bdf8' : '#94a3b8'}
                  fontSize={isSelected ? '12px' : '11px'}
                  fontWeight={isSelected ? '700' : '500'}
                  className="timeline-chart-val"
                >
                  {p.item.value}
                </text>

                {/* Horizon X Axis label */}
                <text
                  x={p.x}
                  y={height - padding.bottom + 18}
                  textAnchor="middle"
                  fill={isSelected ? '#f8fafc' : '#64748b'}
                  fontSize="11px"
                  fontWeight={isSelected ? '700' : '500'}
                >
                  {p.item.horizon}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      {/* Discrete Timeline Values Row */}
      <div className="timeline-horizontal-pills">
        {timeline.map((t) => {
          const isSelected = t.hours === selectedHours;
          return (
            <button
              type="button"
              key={t.horizon}
              className={`timeline-horizon-cell ${isSelected ? 'is-selected' : ''}`}
              onClick={() => onLeadTimeChange && onLeadTimeChange(String(t.hours))}
            >
              <span className="horizon-cell-label">{t.horizon}</span>
              <span className="horizon-cell-val">
                {t.value} <span className="horizon-cell-unit">{unit}</span>
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
