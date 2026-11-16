import React from 'react';
import { clsx } from 'clsx';

export interface CardProps {
  title?: string;
  subtitle?: string;
  headerRight?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  contentClassName?: string;
  highlight?: boolean;
}

export const Card: React.FC<CardProps> = ({
  title,
  subtitle,
  headerRight,
  children,
  className,
  contentClassName,
  highlight = false,
}) => {
  return (
    <div
      className={clsx(
        'bg-surface-card border transition-colors',
        highlight
          ? 'border-brand-cyan/60'
          : 'border-slate-800/80 hover:border-slate-700',
        className
      )}
    >
      {(title || headerRight) && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80">
          <div>
            {title && (
              <h3 className="font-display text-xs font-bold tracking-widest text-slate-200 uppercase">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-[11px] font-mono text-slate-500 mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
          {headerRight && <div className="flex items-center gap-2">{headerRight}</div>}
        </div>
      )}
      <div className={clsx('p-4', contentClassName)}>{children}</div>
    </div>
  );
};
