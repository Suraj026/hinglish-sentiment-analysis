import React from 'react';
import { clsx } from 'clsx';

export interface MetricTileProps {
  label: string;
  value: string | number;
  subValue?: string;
  trend?: {
    value: string;
    positive?: boolean;
  };
  highlight?: boolean;
  icon?: React.ReactNode;
}

export const MetricTile: React.FC<MetricTileProps> = ({
  label,
  value,
  subValue,
  trend,
  highlight = false,
  icon,
}) => {
  return (
    <div
      className={clsx(
        'p-4 bg-surface-card border transition-colors',
        highlight
          ? 'border-brand-cyan/70 bg-surface-elevated/40'
          : 'border-slate-800/80 hover:border-slate-700'
      )}
    >
      <div className="flex items-center justify-between text-slate-400 mb-2">
        <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400">
          {label}
        </span>
        {icon && <span className="text-slate-500">{icon}</span>}
      </div>

      <div className="flex items-baseline gap-2">
        <span
          className={clsx(
            'font-display text-2xl font-bold tracking-tight',
            highlight ? 'text-brand-cyan' : 'text-slate-100'
          )}
        >
          {value}
        </span>
        {subValue && (
          <span className="font-mono text-xs text-slate-400">
            {subValue}
          </span>
        )}
      </div>

      {trend && (
        <div className="mt-2 flex items-center gap-1.5 font-mono text-[11px]">
          <span
            className={
              trend.positive ? 'text-brand-emerald' : 'text-brand-red'
            }
          >
            {trend.positive ? '▲' : '▼'} {trend.value}
          </span>
          <span className="text-slate-400 text-[10px]">vs previous</span>
        </div>
      )}
    </div>
  );
};
