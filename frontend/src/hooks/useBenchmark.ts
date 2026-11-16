import { useQuery } from '@tanstack/react-query';
import { getBenchmark, getModels } from '@/lib/api';
import type { BenchmarkData, ClassMetrics, ConfusionMatrixData, ModelName } from '@/types';

// ─── Static fallback data (from results.csv / classification_report.csv) ──────

const STATIC_CLASS_METRICS: ClassMetrics[] = [
  { label: 'negative', precision: 0.861, recall: 0.884, f1: 0.873, support: 890 },
  { label: 'neutral',  precision: 0.879, recall: 0.854, f1: 0.866, support: 1128 },
  { label: 'positive', precision: 0.871, recall: 0.879, f1: 0.875, support: 982 },
];

const STATIC_CONFUSION_MATRIX: ConfusionMatrixData = {
  labels: ['Negative', 'Neutral', 'Positive'],
  matrix: [
    [787, 55,  48],
    [79,  963, 86],
    [52,  67,  863],
  ],
};

export const STATIC_BENCHMARK: BenchmarkData = {
  models: [
    {
      name: 'bilstm_conv' as ModelName,
      display_name: 'BiLSTM+Conv',
      description: 'Embedding → BiLSTM → Conv1D → Pool → Linear',
      accuracy: 0.9157,
      macro_f1: 0.9164,
      weighted_f1: 0.9156,
      inference_count: 0,
      avg_latency_ms: 0,
    },
    {
      name: 'bilstm' as ModelName,
      display_name: 'BiLSTM',
      description: 'Embedding → BiLSTM → Linear',
      accuracy: 0.9153,
      macro_f1: 0.9161,
      weighted_f1: 0.9153,
      inference_count: 0,
      avg_latency_ms: 0,
    },
    {
      name: 'cnn' as ModelName,
      display_name: 'CNN',
      description: 'Embedding → Conv1D(k=2,3,4) → MaxPool → Linear',
      accuracy: 0.9023,
      macro_f1: 0.9035,
      weighted_f1: 0.9025,
      inference_count: 0,
      avg_latency_ms: 0,
    },
    {
      name: 'lstm' as ModelName,
      display_name: 'LSTM',
      description: 'Embedding → LSTM → Linear',
      accuracy: 0.8987,
      macro_f1: 0.9001,
      weighted_f1: 0.8988,
      inference_count: 0,
      avg_latency_ms: 0,
    },
  ],
  class_metrics: {
    bilstm_conv: STATIC_CLASS_METRICS,
    bilstm:      STATIC_CLASS_METRICS,
    cnn:         STATIC_CLASS_METRICS,
    lstm:        STATIC_CLASS_METRICS,
  } as Record<ModelName, ClassMetrics[]>,
  confusion_matrices: {
    bilstm_conv: STATIC_CONFUSION_MATRIX,
    bilstm:      STATIC_CONFUSION_MATRIX,
    cnn:         STATIC_CONFUSION_MATRIX,
    lstm:        STATIC_CONFUSION_MATRIX,
  } as Record<ModelName, ConfusionMatrixData>,
};

// ─── Hooks ─────────────────────────────────────────────────────────────────────

export function useBenchmark() {
  return useQuery<BenchmarkData>({
    queryKey: ['benchmark'],
    queryFn: async () => {
      try {
        return await getBenchmark();
      } catch {
        return STATIC_BENCHMARK;
      }
    },
    staleTime: 5 * 60 * 1000,
    placeholderData: STATIC_BENCHMARK,
  });
}

export function useModels() {
  return useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    staleTime: 30_000,
  });
}
