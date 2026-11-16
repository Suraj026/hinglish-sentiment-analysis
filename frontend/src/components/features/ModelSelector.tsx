import React from 'react';
import type { ModelName } from '../../types';

interface Model {
  value: ModelName;
  label: string;
  badge?: string;
}

const MODELS: Model[] = [
  { value: 'bilstm_conv', label: 'BiLSTM+Conv', badge: '★ BEST' },
  { value: 'bilstm', label: 'BiLSTM' },
  { value: 'cnn', label: 'CNN' },
  { value: 'lstm', label: 'LSTM' },
];

interface ModelSelectorProps {
  value: ModelName;
  onChange: (v: ModelName) => void;
  disabled?: boolean;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ value, onChange, disabled }) => {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as ModelName)}
        disabled={disabled}
        style={{
          background: '#0a0e17',
          border: '1px solid #3d494c',
          borderRadius: 0,
          color: '#dfe2ef',
          fontFamily: 'Geist, sans-serif',
          fontSize: '12px',
          height: '32px',
          padding: '0 28px 0 10px',
          appearance: 'none',
          WebkitAppearance: 'none',
          cursor: disabled ? 'not-allowed' : 'pointer',
          outline: 'none',
          letterSpacing: '0.04em',
          opacity: disabled ? 0.5 : 1,
          minWidth: '160px',
        }}
        onFocus={(e) => { e.currentTarget.style.borderColor = '#4cd7f6'; }}
        onBlur={(e) => { e.currentTarget.style.borderColor = '#3d494c'; }}
      >
        {MODELS.map((m) => (
          <option key={m.value} value={m.value}>
            {m.label}{m.badge ? `  ${m.badge}` : ''}
          </option>
        ))}
      </select>
      {/* Custom chevron */}
      <div
        style={{
          position: 'absolute',
          right: '8px',
          top: '50%',
          transform: 'translateY(-50%)',
          pointerEvents: 'none',
          color: '#4cd7f6',
          fontSize: '10px',
        }}
      >
        ▾
      </div>
    </div>
  );
};

export default ModelSelector;
