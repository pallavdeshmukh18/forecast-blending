import React from 'react';
import { MapPin, AlertTriangle, Clock, History, ChevronDown } from 'lucide-react';
import { INDIAN_LOCATIONS } from '../../../services/api';

export default function ExtremeFilters({
  selectedLocation,
  onLocationChange,
  selectedHazard,
  onHazardChange,
  selectedLeadTime,
  onLeadTimeChange,
  selectedRun,
  onRunChange,
  isLoading,
}) {
  const hazards = [
    { id: 'heavy_rain', label: 'Heavy Rainfall' },
    { id: 'heatwave', label: 'Heatwave' },
    { id: 'high_wind', label: 'High Wind' },
  ];

  const leadTimes = [
    { id: '6', label: '+6h' },
    { id: '12', label: '+12h' },
    { id: '24', label: '+24h' },
    { id: '48', label: '+48h' },
    { id: '72', label: '+72h' },
  ];

  const runs = [
    { id: 'latest', label: 'Latest (14:00 IST)' },
    { id: 'run-0800', label: 'Run #20260930-0800 (-6h)' },
  ];

  return (
    <div className="forecast-controls-bar extreme-filters-bar">
      {/* 1. Location Selector */}
      <div className="control-group">
        <label className="control-label" htmlFor="ef-location-select">
          <MapPin size={13} className="control-icon" />
          <span>Location</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="ef-location-select"
            value={selectedLocation}
            onChange={(e) => onLocationChange(e.target.value)}
            disabled={isLoading}
            className="control-select"
          >
            {INDIAN_LOCATIONS.map((loc) => (
              <option key={loc.id} value={loc.id}>
                {loc.name}, {loc.state}
              </option>
            ))}
          </select>
          <ChevronDown size={14} className="control-chevron" />
        </div>
      </div>

      {/* 2. Hazard Selector */}
      <div className="control-group">
        <label className="control-label" htmlFor="ef-hazard-select">
          <AlertTriangle size={13} className="control-icon" />
          <span>Hazard</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="ef-hazard-select"
            value={selectedHazard}
            onChange={(e) => onHazardChange(e.target.value)}
            disabled={isLoading}
            className="control-select"
          >
            {hazards.map((h) => (
              <option key={h.id} value={h.id}>
                {h.label}
              </option>
            ))}
          </select>
          <ChevronDown size={14} className="control-chevron" />
        </div>
      </div>

      {/* 3. Forecast Horizon Selector */}
      <div className="control-group">
        <label className="control-label" htmlFor="ef-leadtime-select">
          <Clock size={13} className="control-icon" />
          <span>Forecast Horizon</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="ef-leadtime-select"
            value={selectedLeadTime}
            onChange={(e) => onLeadTimeChange(e.target.value)}
            disabled={isLoading}
            className="control-select"
          >
            {leadTimes.map((lt) => (
              <option key={lt.id} value={lt.id}>
                {lt.label}
              </option>
            ))}
          </select>
          <ChevronDown size={14} className="control-chevron" />
        </div>
      </div>

      {/* 4. Forecast Run */}
      <div className="control-group">
        <label className="control-label" htmlFor="ef-run-select">
          <History size={13} className="control-icon" />
          <span>Forecast Run</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="ef-run-select"
            value={selectedRun}
            onChange={(e) => onRunChange(e.target.value)}
            disabled={isLoading}
            className="control-select"
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.label}
              </option>
            ))}
          </select>
          <ChevronDown size={14} className="control-chevron" />
        </div>
      </div>
    </div>
  );
}
