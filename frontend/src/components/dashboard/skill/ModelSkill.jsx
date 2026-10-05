import React, { useState, useEffect, useCallback } from 'react';
import { RefreshCw, AlertTriangle, Database, CheckCircle2, ShieldCheck } from 'lucide-react';
import SkillFilters from './SkillFilters';
import ModelPerformanceCards from './ModelPerformanceCards';
import ModelComparisonTable from './ModelComparisonTable';
import SkillByLeadTime from './SkillByLeadTime';
import RecentVsHistorical from './RecentVsHistorical';
import SkyCastBlendPerformance from './SkyCastBlendPerformance';
import VerificationMetadata from './VerificationMetadata';
import {
  getModelSkillData,
  generateFallbackModelSkill,
  INDIAN_LOCATIONS,
} from '../../../services/api';

export default function ModelSkill() {
  const [location, setLocation] = useState('mumbai');
  const [variable, setVariable] = useState('rainfall');
  const [leadTime, setLeadTime] = useState('24');
  const [period, setPeriod] = useState('recent');

  const [skillData, setSkillData] = useState(() =>
    generateFallbackModelSkill('mumbai', 'rainfall', 24, 'recent')
  );
  const [isLoading, setIsLoading] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const fetchSkillData = useCallback(async () => {
    setIsLoading(true);
    setHasError(false);

    try {
      const res = await getModelSkillData({
        location,
        variable,
        leadTime,
        period,
      });

      if (res && res.data) {
        setSkillData(res.data);
      } else {
        setHasError(true);
        setErrorMessage('Unable to load model skill data.');
      }
    } catch (err) {
      console.error('[ModelSkill] Error loading data:', err);
      setHasError(true);
      setErrorMessage(err.message || 'Unable to load model skill data.');
    } finally {
      setIsLoading(false);
    }
  }, [location, variable, leadTime, period]);

  useEffect(() => {
    fetchSkillData();
  }, [fetchSkillData]);

  const locObj = INDIAN_LOCATIONS.find((l) => l.id === location) || INDIAN_LOCATIONS[0];
  const locName = `${locObj.name}, ${locObj.state}`;

  return (
    <div className="model-skill-page-container">
      {/* 1. Header */}
      <div className="overview-header skill-header">
        <div className="overview-header-left">
          <h1 className="overview-title">Model Skill</h1>
          <p className="overview-subtitle">
            Compare historical forecasting performance across forecasting systems
          </p>
        </div>

        <div className="overview-header-right">
          <div className="skill-header-telemetry-pill">
            <span className="telemetry-lbl">Last updated:</span>
            <span className="telemetry-time">14:00 IST</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-target">Station: {locObj.name}</span>
            <span className="telemetry-sep">•</span>
            <span className="telemetry-status-live">
              <span className="live-status-dot status-green"></span>
              Verification Engine Active
            </span>
          </div>
        </div>
      </div>

      {/* 2. Filter Bar */}
      <SkillFilters
        selectedLocation={location}
        onLocationChange={setLocation}
        selectedVariable={variable}
        onVariableChange={setVariable}
        selectedLeadTime={leadTime}
        onLeadTimeChange={setLeadTime}
        selectedPeriod={period}
        onPeriodChange={setPeriod}
        isLoading={isLoading}
      />

      {/* 3. Inline Loading Indicator */}
      {isLoading && (
        <div className="forecast-inline-loader">
          <RefreshCw size={14} className="spinner-icon" />
          <span>Refreshing walk-forward verification metrics and error statistics...</span>
        </div>
      )}

      {/* 4. Error State */}
      {hasError ? (
        <div className="forecast-error-box">
          <AlertTriangle size={28} className="error-icon" />
          <div className="error-text-content">
            <h4>Unable to load model skill data.</h4>
            <p>{errorMessage || 'The verification service could not be reached.'}</p>
          </div>
          <button type="button" className="error-retry-btn" onClick={fetchSkillData}>
            <RefreshCw size={14} />
            <span>Retry</span>
          </button>
        </div>
      ) : !skillData || !skillData.models || skillData.models.length === 0 ? (
        /* Empty / No Data State */
        <div className="overview-placeholder-container">
          <div className="placeholder-box">
            <Database size={36} color="#64748b" />
            <h3>No verification data is available for the selected configuration.</h3>
            <p>Try selecting a different location, target variable, or forecast lead time.</p>
            <button className="placeholder-back-btn" onClick={() => { setVariable('rainfall'); setLeadTime('24'); }}>
              <span>Reset Filters</span>
            </button>
          </div>
        </div>
      ) : (
        /* 5. Core Model Skill Content */
        <div className={`skill-content-flow ${isLoading ? 'is-refreshing' : ''}`}>
          {/* Section 3: Model Performance Cards */}
          <section className="skill-section">
            <ModelPerformanceCards
              models={skillData.models}
              unit={skillData.unit}
              periodLabel={skillData.periodLabel}
            />
          </section>

          {/* Section 4: Model Comparison Table */}
          <section className="skill-section">
            <ModelComparisonTable
              models={skillData.models}
              unit={skillData.unit}
            />
          </section>

          {/* Section 5 & 6: Two Column Layout (Skill by Lead Time & Recent vs Historical) */}
          <section className="skill-section two-column-grid">
            <SkillByLeadTime
              leadTimeHorizons={skillData.leadTimeHorizons}
              unit={skillData.unit}
              varName={skillData.varName}
            />
            <RecentVsHistorical
              models={skillData.models}
              unit={skillData.unit}
            />
          </section>

          {/* Section 7: BharatWeather Blend Performance */}
          <section className="skill-section">
            <SkyCastBlendPerformance
              blendPerformance={skillData.blendPerformance}
              unit={skillData.unit}
            />
          </section>

          {/* Section 8: Verification Information */}
          <section className="skill-section">
            <VerificationMetadata
              metadata={skillData.verificationMetadata}
            />
          </section>
        </div>
      )}
    </div>
  );
}
