import React, { useState } from 'react';
import {
  Cloud,
  LogOut,
  LayoutDashboard,
  Compass,
  Cpu,
  BarChart2,
  AlertTriangle,
  HelpCircle,
  GitCompare,
  Clock,
  Server,
  Layers,
  ArrowLeft,
  Map
} from 'lucide-react';
import Overview from './Overview';
import ForecastExplorer from './forecast/ForecastExplorer';
import ModelIntelligence from './intelligence/ModelIntelligence';
import ExtremeWeather from './extremes/ExtremeWeather';
import ModelSkill from './skill/ModelSkill';
import Uncertainty from './uncertainty/Uncertainty';
import ForecastChanges from './changes/ForecastChanges';
import ForecastRuns from './runs/ForecastRuns';
import SystemStatus from './status/SystemStatus';
import WeatherMapWorkspace from './WeatherMapWorkspace';

export default function Dashboard({ user, onLogout }) {
  const [selectedTab, setSelectedTab] = useState(() => {
    if (typeof window !== 'undefined' && window.location.hash.includes('weather-map')) {
      return 'weather-map';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('overview')) {
      return 'overview';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('explorer')) {
      return 'explorer';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('intelligence')) {
      return 'intelligence';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('extremes')) {
      return 'extremes';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('skill')) {
      return 'skill';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('changes')) {
      return 'changes';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('runs')) {
      return 'runs';
    }
    if (typeof window !== 'undefined' && window.location.hash.includes('status')) {
      return 'status';
    }
    return 'weather-map';
  });

  const mainNavItems = [
    { id: 'weather-map', label: 'Weather Map', icon: Map, disabled: false },
    { id: 'overview', label: 'Overview', icon: LayoutDashboard, disabled: false },
    { id: 'explorer', label: 'Forecast Explorer', icon: Compass, disabled: false },
    { id: 'intelligence', label: 'Model Intelligence', icon: Cpu, disabled: false },
    { id: 'skill', label: 'Model Skill', icon: BarChart2, disabled: false },
    { id: 'extremes', label: 'Extreme Weather', icon: AlertTriangle, disabled: false },
    { id: 'uncertainty', label: 'Uncertainty', icon: HelpCircle, disabled: false },
    { id: 'changes', label: 'Forecast Changes', icon: GitCompare, disabled: false },
  ];

  const secondaryNavItems = [
    { id: 'runs', label: 'Forecast Runs', icon: Clock, disabled: false },
    { id: 'status', label: 'System Status', icon: Server, disabled: false },
  ];

  return (
    <div className="dashboard-app-root">
      {/* Top Operational Navigation */}
      <header className="dashboard-topbar">
        <div
          className="d-topbar-left portal-brand-clickable"
          role="link"
          tabIndex={0}
          onClick={onLogout}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onLogout(); } }}
          title="Return to BharatWeather Landing Page"
        >
          <div className="nav-logo-icon" style={{ width: 36, height: 36 }}>
            <Cloud size={20} fill="white" strokeWidth={1.5} />
          </div>
          <div>
            <div className="d-platform-title">BharatWeather Operational Portal</div>
            <div className="d-platform-sub">Hybrid AI–NWP Forecast Blending Platform</div>
          </div>
        </div>

        {/* User Profile & Exit */}
        <div className="d-user-controls">
          <div className="d-user-info">
            <span className="d-user-name">{user?.name || 'Dr. Anuj G.'}</span>
            <span className="d-user-role">{user?.role || 'Lead Meteorologist'}</span>
          </div>
          <button className="d-logout-btn" onClick={onLogout} title="Return to Public Landing Page">
            <LogOut size={15} />
            <span>Exit to Landing Page</span>
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="dashboard-workspace">
        {/* Sidebar Prepared for Full Lifecycle */}
        <aside className="dashboard-sidebar">
          <div className="sidebar-group-title">METEOROLOGICAL INTELLIGENCE</div>
          <div className="sidebar-nav-group">
            {mainNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = selectedTab === item.id;
              return (
                <button
                  key={item.id}
                  className={`sidebar-nav-item ${isActive ? 'active' : ''} ${item.disabled ? 'is-placeholder' : ''}`}
                  onClick={() => setSelectedTab(item.id)}
                  title={item.disabled ? `${item.label} (Scheduled for next release)` : item.label}
                >
                  <Icon size={16} strokeWidth={2} />
                  <span>{item.label}</span>
                  {item.disabled && <span className="nav-pill-soon">Soon</span>}
                </button>
              );
            })}
          </div>

          <div className="sidebar-divider"></div>

          <div className="sidebar-group-title">SYSTEM OPERATIONS</div>
          <div className="sidebar-nav-group">
            {secondaryNavItems.map((item) => {
              const Icon = item.icon;
              const isActive = selectedTab === item.id;
              return (
                <button
                  key={item.id}
                  className={`sidebar-nav-item ${isActive ? 'active' : ''} ${item.disabled ? 'is-placeholder' : ''}`}
                  onClick={() => setSelectedTab(item.id)}
                  title={item.disabled ? `${item.label} (Scheduled for next release)` : item.label}
                >
                  <Icon size={16} strokeWidth={2} />
                  <span>{item.label}</span>
                  {item.disabled && <span className="nav-pill-soon">Soon</span>}
                </button>
              );
            })}
          </div>
        </aside>

        {/* Main Dashboard Screen */}
        <main className={`dashboard-main-content ${selectedTab === 'weather-map' ? 'map-mode' : ''}`}>
          {selectedTab === 'weather-map' ? (
            <div className="dashboard-map-shell">
              <WeatherMapWorkspace onNavigateHome={onLogout} />
            </div>
          ) : selectedTab === 'overview' ? (
            <Overview onNavigateTab={setSelectedTab} />
          ) : selectedTab === 'explorer' ? (
            <ForecastExplorer />
          ) : selectedTab === 'intelligence' ? (
            <ModelIntelligence />
          ) : selectedTab === 'extremes' ? (
            <ExtremeWeather />
          ) : selectedTab === 'skill' ? (
            <ModelSkill />
          ) : selectedTab === 'uncertainty' ? (
            <Uncertainty />
          ) : selectedTab === 'changes' ? (
            <ForecastChanges />
          ) : selectedTab === 'runs' ? (
            <ForecastRuns />
          ) : selectedTab === 'status' ? (
            <SystemStatus />
          ) : (
            <div className="overview-placeholder-container">
              <div className="placeholder-box">
                <Layers size={36} color="#3b82f6" />
                <h3>{mainNavItems.concat(secondaryNavItems).find((i) => i.id === selectedTab)?.label || 'Module'}</h3>
                <p>This operational section is queued for deployment in the next sprint.</p>
                <button className="placeholder-back-btn" onClick={() => setSelectedTab('overview')}>
                  <ArrowLeft size={15} />
                  <span>Return to Overview</span>
                </button>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
