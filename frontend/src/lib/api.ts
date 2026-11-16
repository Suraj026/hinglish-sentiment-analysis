import type {
  AnalyzeRequest,
  BatchAnalyzeRequest,
  BatchResult,
  BenchmarkData,
  HealthResponse,
  ModelInfo,
  SentimentResult,
  StatsResponse,
} from '@/types';

const BASE = '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => 'Unknown error');
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ─── Health ───────────────────────────────────────────────────────────
export const getHealth = (): Promise<HealthResponse> =>
  request<HealthResponse>('/health');

// ─── Models ───────────────────────────────────────────────────────────
export const getModels = (): Promise<ModelInfo[]> =>
  request<ModelInfo[]>('/models');

// ─── Stats ────────────────────────────────────────────────────────────
export const getStats = (): Promise<StatsResponse> =>
  request<StatsResponse>('/stats');

// ─── Analyze ──────────────────────────────────────────────────────────
export const analyzeText = (body: AnalyzeRequest): Promise<SentimentResult> =>
  request<SentimentResult>('/analyze', {
    method: 'POST',
    body: JSON.stringify(body),
  });

export const analyzeBatch = (body: BatchAnalyzeRequest): Promise<BatchResult> =>
  request<BatchResult>('/analyze/batch', {
    method: 'POST',
    body: JSON.stringify(body),
  });

// ─── Benchmark ────────────────────────────────────────────────────────
export const getBenchmark = (): Promise<BenchmarkData> =>
  request<BenchmarkData>('/benchmark');
