import React from 'react';
import { Cloud } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-container">
        <div className="footer-left">
          <div className="nav-logo-icon" style={{ width: '34px', height: '34px' }}>
            <Cloud size={20} fill="white" strokeWidth={1.5} />
          </div>
          <div className="nav-brand-text">
            <span className="brand-name" style={{ fontSize: '18px' }}>BharatWeather</span>
            <span className="brand-tagline">AI for a Safer Tomorrow</span>
          </div>
        </div>

        <nav className="footer-nav">
          <a href="#home">Home</a>
          <a href="#forecast">Forecast</a>
          <a href="#about">About</a>
        </nav>

        <div className="footer-right">
          Smarter Weather.<br />
          A Brighter India.
          <div className="footer-copy">© 2026 BharatWeather. All rights reserved.</div>
        </div>
      </div>
    </footer>
  );
}
