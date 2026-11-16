import React from 'react';
import type { LanguageMix } from '../../types';

interface LanguageMixBadgeProps {
  languageMix: LanguageMix;
}

export const LanguageMixBadge: React.FC<LanguageMixBadgeProps> = ({ languageMix }) => {
  const { hindi_ratio, english_ratio, is_hinglish } = languageMix;

  // Clamp to avoid floating precision issues
  const en = Math.round(english_ratio * 100);
  const hi = Math.round(hindi_ratio * 100);
  const hk = Math.max(0, 100 - en - hi);

  return (
    <div style={{ marginTop: '20px' }}>
      {/* Header */}
      <div
        style={{
          fontFamily: 'Geist Mono, monospace',
          fontSize: '11px',
          fontWeight: 500,
          letterSpacing: '0.12em',
          color: '#4cd7f6',
          textTransform: 'uppercase',
          marginBottom: '12px',
        }}
      >
        LANGUAGE ANALYSIS
      </div>

      {/* Badges row */}
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '12px' }}>
        {is_hinglish && (
          <span
            style={{
              background: 'rgba(76,215,246,0.15)',
              border: '1px solid #4cd7f6',
              color: '#4cd7f6',
              fontFamily: 'Geist Mono, monospace',
              fontSize: '10px',
              letterSpacing: '0.1em',
              padding: '2px 8px',
              fontWeight: 600,
            }}
          >
            HINGLISH
          </span>
        )}
        <span
          style={{
            background: '#1c1f29',
            border: '1px solid #3d494c',
            color: '#4cd7f6',
            fontFamily: 'Geist Mono, monospace',
            fontSize: '10px',
            letterSpacing: '0.08em',
            padding: '2px 8px',
          }}
        >
          EN {en}%
        </span>
        <span
          style={{
            background: '#1c1f29',
            border: '1px solid #3d494c',
            color: '#c0c1ff',
            fontFamily: 'Geist Mono, monospace',
            fontSize: '10px',
            letterSpacing: '0.08em',
            padding: '2px 8px',
          }}
        >
          HI {hi}%
        </span>
        {hk > 0 && (
          <span
            style={{
              background: '#1c1f29',
              border: '1px solid #3d494c',
              color: '#4edea3',
              fontFamily: 'Geist Mono, monospace',
              fontSize: '10px',
              letterSpacing: '0.08em',
              padding: '2px 8px',
            }}
          >
            HK {hk}%
          </span>
        )}
      </div>

      {/* Stacked bar */}
      <div
        style={{
          height: '8px',
          background: '#1c1f29',
          border: '1px solid #3d494c',
          display: 'flex',
          overflow: 'hidden',
        }}
      >
        {/* English — cyan */}
        <div
          style={{
            width: `${en}%`,
            background: '#4cd7f6',
            transition: 'width 0.6s ease',
          }}
        />
        {/* Hindi — indigo */}
        <div
          style={{
            width: `${hi}%`,
            background: '#3131c0',
            transition: 'width 0.6s ease',
          }}
        />
        {/* Hinglish keywords — emerald */}
        {hk > 0 && (
          <div
            style={{
              flex: 1,
              background: '#4edea3',
              transition: 'width 0.6s ease',
            }}
          />
        )}
      </div>

      {/* Labels */}
      <div
        style={{
          display: 'flex',
          gap: '12px',
          marginTop: '8px',
        }}
      >
        {[
          { label: 'English', color: '#4cd7f6' },
          { label: 'Hindi', color: '#c0c1ff' },
          ...(hk > 0 ? [{ label: 'Hinglish KW', color: '#4edea3' }] : []),
        ].map(({ label, color }) => (
          <div
            key={label}
            style={{ display: 'flex', alignItems: 'center', gap: '5px' }}
          >
            <div style={{ width: '8px', height: '8px', background: color }} />
            <span
              style={{
                fontFamily: 'Geist Mono, monospace',
                fontSize: '10px',
                color: '#869397',
                letterSpacing: '0.05em',
              }}
            >
              {label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

export default LanguageMixBadge;
