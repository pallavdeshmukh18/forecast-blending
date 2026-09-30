import React from 'react';
import { ArrowRight } from 'lucide-react';

export default function FooterCta({ onExploreClick }) {
  return (
    <section className="bottom-cta-section">
      <img
        src="/hero-mountains.jpg"
        alt="A Safer and More Resilient India"
        className="bottom-cta-bg"
      />
      <div className="bottom-cta-overlay"></div>

      <div className="container bottom-cta-content">
        <span className="bottom-cta-eyebrow">BE PREPARED, STAY AHEAD</span>
        <h2 className="bottom-cta-title">A Safer and More Resilient India</h2>
        <p className="bottom-cta-desc">
          Make smarter decisions with intelligent weather forecasts.
        </p>
        <a href="#forecast" onClick={onExploreClick} className="btn-primary">
          Explore Forecast <ArrowRight size={16} />
        </a>
      </div>
    </section>
  );
}
