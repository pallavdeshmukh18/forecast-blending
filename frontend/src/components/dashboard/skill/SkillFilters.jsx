import React from 'react';
import { MapPin, Activity, Clock, Calendar, ChevronDown } from 'lucide-react';
import { INDIAN_LOCATIONS } from '../../../services/api';

export default function SkillFilters({
  selectedLocation,
  onLocationChange,
  selectedVariable,
  onVariableChange,
  selectedLeadTime,
  onLeadTimeChange,
  selectedPeriod,
  onPeriodChange,
  isLoading,
}) {
  const variables = [
    { id: 'rainfall', label: 'Rainfall' },
    { id: 'temperature', label: 'Temperature' },
    { id: 'wind', label: 'Wind Speed' },
  ];

  const leadTimes = [
    { id: '24', label: '+24h' },
    { id: '48', label: '+48h' },
    { id: '72', label: '+72h' },
  ];

  const periods = [
    { id: 'recent', label: 'Recent (Rolling 30 Days)' },
    { id: 'historical', label: 'Historical (Full Season Benchmark)' },
  ];

  return (
    <div className="forecast-controls-bar skill-filters-bar">
      {/* 1. Location Selector */}
      <div className="control-group">
        <label className="control-label" htmlFor="sf-location-select">
          <MapPin size={13} className="control-icon" />
          <span>Location</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="sf-location-select"
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

      {/* 2. Variable Selector */}
      <div className="control-group">
        <label className="control-label" htmlFor="sf-variable-select">
          <Activity size={13} className="control-icon" />
          <span>Variable</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="sf-variable-select"
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

      {/* 3. Lead Time Selector */}
      <div className="control-group">
        <label className="control-label" htmlFor="sf-leadtime-select">
          <Clock size={13} className="control-icon" />
          <span>Lead Time</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="sf-leadtime-select"
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

      {/* 4. Evaluation Period */}
      <div className="control-group">
        <label className="control-label" htmlFor="sf-period-select">
          <Calendar size={13} className="control-icon" />
          <span>Evaluation Period</span>
        </label>
        <div className="control-select-wrap">
          <select
            id="sf-period-select"
            value={selectedPeriod}
            onChange={(e) => onPeriodChange(e.target.value)}
            disabled={isLoading}
            className="control-select"
          >
            {periods.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
          <ChevronDown size={14} className="control-chevron" />
        </div>
      </div>
    </div>
  );
}
