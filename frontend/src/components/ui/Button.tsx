import React from 'react';
import { clsx } from 'clsx';

export type ButtonVariant = 'primary' | 'secondary' | 'destructive' | 'ghost';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  className,
  disabled,
  ...props
}) => {
  const baseStyles =
    'inline-flex items-center justify-center font-mono font-bold tracking-wider uppercase transition-colors select-none focus:outline-none focus:ring-1 focus:ring-brand-cyan disabled:opacity-50 disabled:cursor-not-allowed';

  const variantStyles: Record<ButtonVariant, string> = {
    primary: 'bg-brand-cyan text-brand-black hover:bg-brand-cyan-bright border-0',
    secondary: 'bg-transparent text-slate-200 border border-slate-700 hover:border-slate-500 hover:bg-surface-elevated',
    destructive: 'bg-transparent text-brand-red border border-brand-red hover:bg-brand-red hover:text-brand-black',
    ghost: 'bg-transparent text-slate-400 hover:text-slate-100 hover:bg-surface-card border-0',
  };

  const sizeStyles = {
    sm: 'h-7 px-3 text-xs',
    md: 'h-8 px-4 text-xs',
    lg: 'h-10 px-6 text-sm',
  };

  return (
    <button
      className={clsx(baseStyles, variantStyles[variant], sizeStyles[size], className)}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <span className="flex items-center gap-2">
          <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
          </svg>
          PROCESSING
        </span>
      ) : (
        children
      )}
    </button>
  );
};
