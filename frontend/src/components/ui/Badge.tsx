import React from 'react';
import { clsx } from 'clsx';
import { Sentiment } from '../../types';

export type BadgeVariant = 'positive' | 'negative' | 'neutral' | 'telemetry' | 'secondary' | 'champion';

export interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  sentiment?: Sentiment;
  score?: number | string;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant,
  sentiment,
  score,
  className,
}) => {
  const resolvedVariant: BadgeVariant =
    variant ||
    (sentiment === 'positive'
      ? 'positive'
      : sentiment === 'negative'
      ? 'negative'
      : sentiment === 'neutral'
      ? 'neutral'
      : 'secondary');

  const variantStyles: Record<BadgeVariant, string> = {
    positive: 'bg-[#06281e] border border-brand-emerald text-brand-emerald',
    negative: 'bg-[#2d1214] border border-brand-red text-brand-red',
    neutral: 'bg-[#241804] border border-brand-amber text-brand-amber',
    telemetry: 'bg-surface-elevated border border-brand-cyan text-brand-cyan',
    secondary: 'bg-surface-elevated border border-slate-700 text-slate-300',
    champion: 'bg-brand-cyan text-brand-black border-0 font-bold',
  };

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-mono tracking-wider uppercase select-none',
        variantStyles[resolvedVariant],
        className
      )}
    >
      {score !== undefined && (
        <span className="opacity-80 font-semibold">{score}</span>
      )}
      <span>{children}</span>
    </span>
  );
};
