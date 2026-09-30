import React from 'react';
import { MapPin, Activity, Clock, History, GitCompare, ChevronDown } from 'lucide-react';
import { INDIAN_LOCATIONS } from '../../../services/api';

export default function ForecastChangesFilters({
  selectedLocation,
  onLocationChange,
  selectedVariable,
  onVariableChange,
  selectedLeadTime,
  onLeadTimeChange,
  selectedCurrentRun,
  onCurrentRunChange,
  selectedCompareRun,
  onCompareRunChange,
  isLoading,
}) {
  const variables = [
    { id: 'rainfall', label: 'Rainfall' },
    { id: 'temperature', label: 'Temperature' },
    { id: 'wind', label: 'Wind Speed' },
  ];

  const leadTimes = [
    { id: '6', label: '+6h' },
    { id: '12', label: '+12h' },
    { id: '24', label: '+24h' },
    { id: '48', label: '+48h' },
    { id: '72', label: '+72h' },
  ];

  const runs = [
    { id: 'run-1400', label: 'Run #20260930-1400 (Latest)' },
    { id: 'run-0800', label: 'Run #20260930-0800 (-6h)' },
    { id: 'run-0200', label: 'Run #20260930-0200 (-12h)' },
  ];

  return (
    <div className="forecast-controls-bar changes-filters-bar">
      {/* 1. Current Run */}
      <div className="control-group">
        <label className="control-label" htmlFor="cf-current-run">
          <History size={13} className="control-icon" />
          <span>Current Run</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="cf-current-run"
            value={selectedCurrentRun}
            onChange={(e) => onCurrentRunChange(e.target.value)}
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

      {/* 2. Compare With Run */}
      <div className="control-group">
        <label className="control-label" htmlFor="cf-compare-run">
          <GitCompare size={13} className="control-icon" />
          <span>Compare With</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="cf-compare-run"
            value={selectedCompareRun}
            onChange={(e) => onCompareRunChange(e.target.value)}
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

      {/* 3. Location */}
      <div className="control-group">
        <label className="control-label" htmlFor="cf-location-select">
          <MapPin size={13} className="control-icon" />
          <span>Location</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="cf-location-select"
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

      {/* 4. Variable */}
      <div className="control-group">
        <label className="control-label" htmlFor="cf-variable-select">
          <Activity size={13} className="control-icon" />
          <span>Variable</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="cf-variable-select"
            value={selectedVariable}
            onChange={(e) => onVariableChange(e.target.value)}
            disabled={isLoading}
            className="control-select"
          >
            {variables.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
          <ChevronDown size={14} className="control-chevron" />
        </div>
      </div>

      {/* 5. Forecast Horizon */}
      <div className="control-group">
        <label className="control-label" htmlFor="cf-leadtime-select">
          <Clock size={13} className="control-icon" />
          <span>Horizon</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="cf-leadtime-select"
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
    </div>
  );
}
