import React, { useState, useEffect, useCallback } from 'react';
import { Cpu, RefreshCw, ChevronDown, MapPin, Activity, Clock } from 'lucide-react';
import BlendSummaryCard from './BlendSummaryCard';
import ModelContributionCard from './ModelContributionCard';
import AttributionExplainabilityCard from './AttributionExplainabilityCard';
import ModelPerformanceCard from './ModelPerformanceCard';
import ModelForecastsMiniCard from './ModelForecastsMiniCard';
import { getModelIntelligenceData, INDIAN_LOCATIONS } from '../../../services/api';

export default function ModelIntelligence() {
  const [location, setLocation] = useState('mumbai');
  const [variable, setVariable] = useState('rainfall');
  const [leadTime, setLeadTime] = useState('24');

  const [intelligenceData, setIntelligenceData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const fetchIntelligence = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await getModelIntelligenceData({
        location,
        variable,
        leadTime,
      });
      if (res && res.data) {
        setIntelligenceData(res.data);
      }
    } catch (err) {
      console.error('[ModelIntelligence] Error fetching data:', err);
    } finally {
      setIsLoading(false);
    }
  }, [location, variable, leadTime]);

  useEffect(() => {
    fetchIntelligence();
  }, [fetchIntelligence]);

  const locObj = INDIAN_LOCATIONS.find((l) => l.id === location) || INDIAN_LOCATIONS[0];
  const varLabel = variable === 'rainfall' ? 'Rainfall' : variable === 'temperature' ? 'Temperature' : 'Wind';

  return (
    <div className="model-intelligence-container">
      {/* 1. Header */}
      <div className="overview-header intelligence-header">
        <div className="overview-header-left">
          <h1 className="overview-title">MODEL INTELLIGENCE</h1>
          <p className="overview-subtitle">Why SkyCast made this forecast</p>
        </div>

        <div className="overview-header-right">
          <div className="intelligence-engine-pill">
            <Cpu size={14} className="engine-icon" />
            <span>Regime-Conditioned Bayesian Blender</span>
          </div>
        </div>
      </div>

      {/* 2. Interactive Context Selector Bar */}
      <div className="intelligence-context-bar">
        <div className="context-selection-summary">
          <span className="context-prefix">Selected:</span>
          <span className="context-token highlighted-city">{locObj.name}</span>
          <span className="context-sep">|</span>
          <span className="context-token highlighted-var">{varLabel}</span>
          <span className="context-sep">|</span>
          <span className="context-token highlighted-horizon">+{leadTime}h</span>
        </div>

        <div className="context-selectors-group">
          {/* Location Dropdown */}
          <div className="mini-select-wrap">
            <MapPin size={12} className="mini-select-icon" />
            <select
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="mini-select"
              aria-label="Location"
            >
              {INDIAN_LOCATIONS.map((loc) => (
                <option key={loc.id} value={loc.id}>
                  {loc.name}
                </option>
              ))}
            </select>
            <ChevronDown size={12} className="mini-select-chevron" />
          </div>

          {/* Variable Dropdown */}
          <div className="mini-select-wrap">
            <Activity size={12} className="mini-select-icon" />
            <select
              value={variable}
              onChange={(e) => setVariable(e.target.value)}
              className="mini-select"
              aria-label="Variable"
            >
              <option value="rainfall">Rainfall</option>
              <option value="temperature">Temperature</option>
              <option value="wind">Wind</option>
            </select>
            <ChevronDown size={12} className="mini-select-chevron" />
          </div>

          {/* Lead Time Dropdown */}
          <div className="mini-select-wrap">
            <Clock size={12} className="mini-select-icon" />
            <select
              value={leadTime}
              onChange={(e) => setLeadTime(e.target.value)}
              className="mini-select"
              aria-label="Lead Time"
            >
              <option value="6">+6h</option>
              <option value="12">+12h</option>
              <option value="24">+24h</option>
              <option value="48">+48h</option>
              <option value="72">+72h</option>
            </select>
            <ChevronDown size={12} className="mini-select-chevron" />
          </div>
        </div>
      </div>

      {/* 3. Inline Loading Indicator */}
      {isLoading && (
        <div className="forecast-inline-loader">
          <RefreshCw size={14} className="spinner-icon" />
          <span>Updating attribution weights and verification telemetry...</span>
        </div>
      )}

      {/* 4. Core Content Grids */}
      <div className="intelligence-content-flow">
        {/* Upper Row: Blend & Contribution (Left) + Why These Weights? (Right) */}
        <div className="intelligence-two-col-grid">
          <div className="left-stack-group">
            <BlendSummaryCard data={intelligenceData} />
            <ModelContributionCard contributions={intelligenceData?.contributions} />
          </div>

          <div className="right-stack-group">
            <AttributionExplainabilityCard factors={intelligenceData?.attributionFactors} />
          </div>
        </div>

        {/* Lower Row: Model Performance (Left) + Model Forecasts (Right) */}
        <div className="intelligence-two-col-grid">
          <ModelPerformanceCard
            performance={intelligenceData?.performance}
            variableLabel={varLabel}
          />
          <ModelForecastsMiniCard
            forecastsData={intelligenceData?.forecasts}
            unit={intelligenceData?.unit}
          />
        </div>
      </div>
    </div>
  );
}
