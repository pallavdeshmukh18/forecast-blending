import React, { useState } from 'react';
import { Cloud, Lock, Mail, ArrowRight, X, Shield, Sparkles } from 'lucide-react';

export default function LoginModal({ isOpen, onClose, onLoginSuccess }) {
  const [email, setEmail] = useState('meteorologist@skycast.ai');
  const [password, setPassword] = useState('••••••••••••');
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      onLoginSuccess({
        name: 'Dr. Anuj G.',
        role: 'Chief Research Meteorologist',
        email
      });
    }, 600);
  };

  const handleInstantDemoLogin = () => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
      onLoginSuccess({
        name: 'Dr. Anuj G.',
        role: 'Senior Research Meteorologist',
        email: 'meteorologist@skycast.ai'
      });
    }, 400);
  };

  return (
    <div className="login-modal-backdrop" onClick={onClose}>
      <div className="login-modal-card" onClick={(e) => e.stopPropagation()}>
        {/* Close Button */}
        <button className="login-modal-close" onClick={onClose} aria-label="Close">
          <X size={20} />
        </button>

        {/* Modal Header */}
        <div className="login-modal-header">
          <div className="login-modal-logo">
            <Cloud size={24} fill="white" strokeWidth={1.5} />
          </div>
          <h2 className="login-modal-title">Sign in to SkyCast</h2>
          <p className="login-modal-subtitle">
            Access the Hybrid AI–NWP Forecast Blending &amp; Verification Platform
          </p>
        </div>

        {/* Instant Access Banner */}
        <div className="login-instant-banner" onClick={handleInstantDemoLogin}>
          <div className="instant-badge">
            <Sparkles size={16} color="#2563eb" />
            <span>Fast Access</span>
          </div>
          <div className="instant-content">
            <h4>Launch Dashboard as Meteorologist</h4>
            <p>One-click instant login to explore the operational blending portal</p>
          </div>
          <ArrowRight size={18} className="instant-arrow" />
        </div>

        <div className="login-divider">
          <span>or sign in with credentials</span>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label className="form-label">Email Address</label>
            <div className="input-with-icon">
              <Mail size={16} className="input-icon" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@agency.gov.in"
                className="form-input"
              />
            </div>
          </div>

          <div className="form-group">
            <div className="label-row">
              <label className="form-label">Password</label>
              <a href="#forgot" onClick={(e) => e.preventDefault()} className="forgot-link">
                Forgot?
              </a>
            </div>
            <div className="input-with-icon">
              <Lock size={16} className="input-icon" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="form-input"
              />
            </div>
          </div>

          <button type="submit" className="login-submit-btn" disabled={loading}>
            {loading ? 'Authenticating...' : 'Sign In to Dashboard'}
            {!loading && <ArrowRight size={16} />}
          </button>
        </form>

        <div className="login-modal-footer">
          <Shield size={14} className="shield-icon" />
          <span>Secured with Multi-Model Verification Token System</span>
        </div>
      </div>
    </div>
  );
}
