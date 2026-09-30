import React from 'react';
import { 
  ArrowRight, 
  Globe, 
  MapPin, 
  ShieldCheck, 
  CloudRain, 
  Wind, 
  Droplets,
  CloudDrizzle
} from 'lucide-react';

export default function Hero({ weatherData, onExploreClick }) {
  return (
    <section id="home" className="hero-section">
      {/* Background Landscape with soft atmospheric white gradient */}
      <div className="hero-bg-wrapper">
        <img 
          src="/hero-mountains.jpg" 
          alt="Scenic Western Ghats India landscape" 
          className="hero-bg-image" 
        />
        <div className="hero-gradient-overlay"></div>
        <div className="hero-bottom-fade"></div>
      </div>

      {/* Handwritten / Script Note */}
      <div className="hero-handwritten-note">
        <span>Better forecasts</span>
        <span>A safer tomorrow</span>
      </div>

      <div className="container hero-container">
        {/* Left Column Content */}
        <div className="hero-content">
          <div className="hero-eyebrow">
            ACCURATE &nbsp;·&nbsp; INTELLIGENT &nbsp;·&nbsp; LOCATION-SPECIFIC
          </div>

          <h1 className="hero-title">
            Smarter<br />
            Weather Forecasts<br />
            for a <span className="gradient-text">Brighter India</span>
          </h1>

          <p className="hero-description">
            AI-powered forecasts that blend global weather models to deliver
            more accurate, reliable and easy-to-understand information for a safer
            and more resilient India.
          </p>

          <div className="hero-cta-group">
            <a href="#forecast" onClick={onExploreClick} className="btn-primary">
              Explore Forecast <ArrowRight size={16} />
            </a>
            <a href="#about" className="btn-secondary">
              Learn More
            </a>
          </div>

          <div className="hero-badges-row">
            <div className="hero-badge-item">
              <Globe className="hero-badge-icon" />
              <span>Multiple Global Models</span>
            </div>
            <div className="hero-badge-item">
              <MapPin className="hero-badge-icon" />
              <span>India Focused</span>
            </div>
            <div className="hero-badge-item">
              <ShieldCheck className="hero-badge-icon" />
              <span>Reliable &amp; Explainable</span>
            </div>
          </div>
        </div>

        {/* Right Column: Floating Live Weather Card */}
        <div className="hero-weather-card-container">
          <div className="hero-weather-card">
            <div className="hw-header">
              <span className="hw-location">{weatherData.city}, {weatherData.state}</span>
              <span className="hw-time">{weatherData.timeString || 'Mon, 15 Jul 2024, 6:00 AM'}</span>
            </div>

            <div className="hw-body">
              <div className="hw-main-temp">
                <div className="hw-weather-icon-temp">
                  <CloudDrizzle className="hw-cloud-icon" strokeWidth={1.8} />
                  <span className="hw-temp-value">{weatherData.temp}°C</span>
                </div>
                <span className="hw-condition">{weatherData.condition}</span>
              </div>

              <div className="hw-metrics-list">
                <div className="hw-metric-row">
                  <CloudRain className="hw-metric-icon" />
                  <div className="hw-metric-details">
                    <span className="hw-metric-val">{weatherData.rainfall} mm</span>
                    <span className="hw-metric-lbl">Rainfall (Next 24h)</span>
                  </div>
                </div>

                <div className="hw-metric-row">
                  <Wind className="hw-metric-icon" />
                  <div className="hw-metric-details">
                    <span className="hw-metric-val">{weatherData.wind} km/h</span>
                    <span className="hw-metric-lbl">Wind Speed</span>
                  </div>
                </div>

                <div className="hw-metric-row">
                  <Droplets className="hw-metric-icon" />
                  <div className="hw-metric-details">
                    <span className="hw-metric-val">{weatherData.humidity}%</span>
                    <span className="hw-metric-lbl">Humidity</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
