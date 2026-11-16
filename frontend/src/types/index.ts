// ─── Sentiment ───────────────────────────────────────────────────────
export type Sentiment = 'positive' | 'neutral' | 'negative';

export interface SentimentScores {
  negative: number;
  neutral: number;
  positive: number;
}

export interface LanguageMix {
  hindi_ratio: number;
  english_ratio: number;
  is_hinglish: boolean;
}

export interface SentimentResult {
  text: string;
  model_name: string;
  sentiment: Sentiment;
  confidence: number;
  scores: SentimentScores;
  processing_time_ms: number;
  language_mix: LanguageMix;
  tokens: string[];
}

export interface BatchResult {
  results: SentimentResult[];
  total_time_ms: number;
}

// ─── Models ──────────────────────────────────────────────────────────
export type ModelName = 'lstm' | 'bilstm' | 'cnn' | 'bilstm_conv';

export interface ModelInfo {
  name: ModelName;
  display_name: string;
  description: string;
  accuracy: number;
  macro_f1: number;
  weighted_f1: number;
  inference_count: number;
  avg_latency_ms: number;
}

// ─── Benchmark / Metrics ─────────────────────────────────────────────
export interface ClassMetrics {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface BenchmarkData {
  models: ModelInfo[];
  class_metrics: Record<ModelName, ClassMetrics[]>;
  confusion_matrices: Record<ModelName, ConfusionMatrixData>;
}

// ─── Stats ───────────────────────────────────────────────────────────
export interface ModelDistribution {
  model: ModelName;
  count: number;
  percentage: number;
}

export interface StatsResponse {
  total_analyzed: number;
  avg_latency_ms: number;
  model_distribution: ModelDistribution[];
  sentiment_distribution: {
    positive: number;
    neutral: number;
    negative: number;
  };
  recent_analyses: SentimentResult[];
  uptime_seconds: number;
}

// ─── API ─────────────────────────────────────────────────────────────
export interface AnalyzeRequest {
  text: string;
  model_name?: ModelName;
}

export interface BatchAnalyzeRequest {
  texts: string[];
  model_name?: ModelName;
}

export interface HealthResponse {
  status: string;
  models_loaded: boolean;
  model_count: number;
}

// ─── UI State ────────────────────────────────────────────────────────
export interface ApiError {
  message: string;
  status?: number;
}
