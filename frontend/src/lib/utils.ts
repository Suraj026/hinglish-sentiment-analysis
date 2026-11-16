import type { ModelName, Sentiment } from '@/types';
import { clsx, type ClassValue } from 'clsx';

// ─── clsx re-export ───────────────────────────────────────────────────
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}

// ─── Number formatters ────────────────────────────────────────────────
export function fmtPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function fmtMs(ms: number): string {
  if (ms < 1) return '<1ms';
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function fmtCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function fmtScore(value: number, decimals = 4): string {
  return value.toFixed(decimals);
}

export function fmtDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

// ─── Model helpers ────────────────────────────────────────────────────
export const MODEL_DISPLAY_NAMES: Record<ModelName, string> = {
  lstm: 'LSTM',
  bilstm: 'BiLSTM',
  cnn: 'CNN',
  bilstm_conv: 'BiLSTM+Conv',
};

export function getModelDisplayName(name: ModelName): string {
  return MODEL_DISPLAY_NAMES[name] ?? name;
}

// ─── Sentiment helpers ────────────────────────────────────────────────
export const SENTIMENT_COLORS: Record<Sentiment, { bg: string; border: string; text: string }> = {
  positive: {
    bg: '#06281e',
    border: '#10b981',
    text: '#10b981',
  },
  negative: {
    bg: '#2d1214',
    border: '#ef4444',
    text: '#ef4444',
  },
  neutral: {
    bg: '#241804',
    border: '#f59e0b',
    text: '#f59e0b',
  },
};

export function getSentimentColor(sentiment: Sentiment): { bg: string; border: string; text: string } {
  return SENTIMENT_COLORS[sentiment];
}

export function sentimentLabel(s: Sentiment): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ─── Language helpers ─────────────────────────────────────────────────
export function languageLabel(hindiRatio: number, englishRatio: number): string {
  if (hindiRatio > 0.6) return 'Hindi';
  if (englishRatio > 0.6) return 'English';
  return 'Hinglish';
}

// ─── Misc ─────────────────────────────────────────────────────────────
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function truncateText(text: string, maxLength = 60): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trimEnd() + '…';
}

export function timestampNow(): string {
  return new Date().toISOString();
}
