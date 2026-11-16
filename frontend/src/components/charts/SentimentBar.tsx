import React from 'react';
import { clsx } from 'clsx';
import { Sentiment } from '../../types';

export interface SentimentBarProps {
  label: string;
  score: number;
  sentiment?: Sentiment;
  showPercentage?: boolean;
  className?: string;
}

export const SentimentBar: React.FC<SentimentBarProps> = ({
  label,
  score,
  sentiment,
  showPercentage = true,
  className,
}) => {
  const percentage = Math.min(100, Math.max(0, Math.round(score * 100)));

  const barColor =
    sentiment === 'positive'
      ? 'bg-brand-emerald'
      : sentiment === 'negative'
      ? 'bg-brand-red'
      : sentiment === 'neutral'
      ? 'bg-brand-amber'
      : 'bg-brand-cyan';

  return (
    <div className={clsx('space-y-1 font-mono text-xs', className)}>
      <div className="flex justify-between items-center text-slate-300 text-[11px]">
        <span className="uppercase tracking-wider">{label}</span>
        {showPercentage && (
          <span className="font-bold text-slate-100">{percentage}%</span>
        )}
      </div>
      <div className="h-2 w-full bg-slate-900 border border-slate-800 overflow-hidden">
        <div
          className={clsx('h-full transition-all duration-300', barColor)}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
