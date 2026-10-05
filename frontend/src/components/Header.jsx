import React, { useState } from 'react';
import { Cloud, Search, ArrowRight, LogIn } from 'lucide-react';

export default function Header({ onSearchLocation, currentLocation, onLogin }) {
  const [searchInput, setSearchInput] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (searchInput.trim()) {
      onSearchLocation(searchInput.trim());
      setSearchInput('');
    }
  };

  return (
    <header className="navbar">
      <div className="container nav-container">
        <a href="#home" className="nav-left">
          <div className="nav-logo-icon">
            <Cloud size={22} fill="white" strokeWidth={1.5} />
          </div>
          <div className="nav-brand-text">
            <span className="brand-name">BharatWeather</span>
            <span className="brand-tagline">AI for a Safer Tomorrow</span>
          </div>
        </a>

        <nav className="nav-links">
          <a href="#home" className="nav-link active">Home</a>
          <a
            href="#dashboard"
            onClick={(e) => {
              e.preventDefault();
              onLogin();
            }}
            className="nav-link"
          >
            Dashboard
          </a>
          <a href="#about" className="nav-link">About</a>
        </nav>

        <div className="nav-right-actions">
          <form onSubmit={handleSubmit} className="nav-search">
            <Search className="nav-search-icon" />
            <input
              type="text"
              className="nav-search-input"
              placeholder="Search location (e.g., Mumbai)"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
            <button type="submit" className="nav-search-btn" aria-label="Search">
              <ArrowRight size={14} strokeWidth={2.5} />
            </button>
          </form>

          <button
            type="button"
            className="nav-login-btn"
            onClick={onLogin}
            title="Open Operational Forecast Dashboard"
          >
            <LogIn size={16} />
            <span>Login</span>
          </button>
        </div>
      </div>
    </header>
  );
}
