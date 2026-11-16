import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { analyzeText, analyzeBatch } from '../lib/api';
import type { AnalyzeRequest, BatchAnalyzeRequest, SentimentResult } from '../types';

export function useAnalyze() {
  const [result, setResult] = useState<SentimentResult | null>(null);
  const [history, setHistory] = useState<SentimentResult[]>([]);

  const mutation = useMutation({
    mutationFn: (req: AnalyzeRequest) => analyzeText(req),
    onSuccess: (data: SentimentResult) => {
      setResult(data);
      setHistory((prev) => [data, ...prev.slice(0, 19)]);
    },
  });

  return {
    analyze: (req: AnalyzeRequest) => mutation.mutate(req),
    result,
    isLoading: mutation.isPending,
    error: mutation.error,
    history,
    clearHistory: () => setHistory([]),
  };
}

export function useAnalyzeBatch() {
  return useMutation({
    mutationFn: (req: BatchAnalyzeRequest) => analyzeBatch(req),
  });
}
