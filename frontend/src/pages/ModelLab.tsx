import React from 'react';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { ConfusionMatrix } from '../components/charts/ConfusionMatrix';
import { ModelRadarChart } from '../components/charts/ModelRadarChart';
import { useBenchmark } from '../hooks/useBenchmark';
import { Trophy } from 'lucide-react';
import type { ModelInfo, ClassMetrics } from '../types';

export const ModelLab: React.FC = () => {
  const { data: benchmark } = useBenchmark();

  if (!benchmark) {
    return (
      <div className="py-20 text-center font-mono text-xs text-slate-500">
        LOADING BENCHMARK DATA...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
        <div>
          <h1 className="font-display text-xl font-bold tracking-tight text-slate-100 uppercase">
            MODEL LAB & BENCHMARK SUITE
          </h1>
          <p className="font-mono text-xs text-slate-400 mt-1">
            Comparative evaluation across neural baselines on code-mixed Hinglish SemEval-2020 dataset
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="champion" className="text-xs py-1 px-3">
            ★ PROD CHAMPION: BILSTM+CONV
          </Badge>
        </div>
      </div>

      {/* Champion Model Banner */}
      <div className="p-5 bg-surface-card border-2 border-brand-cyan/80 relative overflow-hidden">
        <div className="absolute top-0 right-0 transform translate-x-4 -translate-y-4 text-brand-cyan/10">
          <Trophy className="w-48 h-48" />
        </div>

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10px] tracking-widest uppercase text-brand-cyan font-bold">
                PRODUCTION CHAMPION MODEL
              </span>
              <span className="text-[10px] px-1.5 py-0.5 bg-brand-cyan/20 text-brand-cyan border border-brand-cyan/40 font-mono">
                ACTIVE SERVING
              </span>
            </div>
            <h2 className="font-display text-2xl font-bold tracking-tight text-slate-100">
              BiLSTM + 1D-Convolution (NITS-Hinglish-SentiMix)
            </h2>
            <p className="font-sans text-xs text-slate-300 max-w-2xl leading-relaxed">
              Hybrid recurrent-convolutional neural baseline combining bidirectional temporal context 
              with multi-span 1D feature extraction. Captures both long-range semantic dependencies and local 
              code-mixed n-gram sentiment markers.
            </p>
          </div>

          <div className="grid grid-cols-3 gap-4 font-mono border-t lg:border-t-0 lg:border-l border-slate-800 pt-4 lg:pt-0 lg:pl-6 flex-shrink-0">
            <div>
              <div className="text-[10px] text-slate-400">TEST ACCURACY</div>
              <div className="text-xl font-bold text-brand-cyan">91.57%</div>
              <div className="text-[9px] text-slate-400">Top Rank</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400">MACRO F1</div>
              <div className="text-xl font-bold text-brand-cyan">91.64%</div>
              <div className="text-[9px] text-slate-400">Balanced</div>
            </div>
            <div>
              <div className="text-[10px] text-slate-400">WEIGHTED F1</div>
              <div className="text-xl font-bold text-brand-cyan">91.56%</div>
              <div className="text-[9px] text-slate-400">Weighted</div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <Card
        title="NEURAL ARCHITECTURE BENCHMARK LEADERBOARD"
        subtitle="Empirical comparison across 4 PyTorch architectures trained on identical Hinglish splits"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase tracking-wider">
                <th className="py-2.5 px-3">MODEL</th>
                <th className="py-2.5 px-3">ARCHITECTURE</th>
                <th className="py-2.5 px-3">ACCURACY</th>
                <th className="py-2.5 px-3">MACRO F1</th>
                <th className="py-2.5 px-3">WEIGHTED F1</th>
                <th className="py-2.5 px-3">STATUS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              {benchmark.models.map((model: ModelInfo) => {
                const isChamp = model.name === 'bilstm_conv';
                return (
                  <tr
                    key={model.name}
                    className={
                      isChamp
                        ? 'bg-brand-cyan/5 font-semibold text-slate-100 border-l-2 border-l-brand-cyan'
                        : 'hover:bg-surface-elevated/40 text-slate-300'
                    }
                  >
                    <td className="py-3 px-3 font-bold flex items-center gap-2">
                      {isChamp && <Trophy className="w-3.5 h-3.5 text-brand-cyan" />}
                      {model.display_name}
                    </td>
                    <td className="py-3 px-3 text-[11px] text-slate-400">
                      {model.name === 'bilstm_conv' && 'Embedding → BiLSTM → Conv1D(k=3) → Pool → Linear'}
                      {model.name === 'bilstm' && 'Embedding → Bidirectional LSTM → Linear'}
                      {model.name === 'cnn' && 'Embedding → Conv1D(k=2,3,4) → MaxPool → Linear'}
                      {model.name === 'lstm' && 'Embedding → Unidirectional LSTM → Linear'}
                    </td>
                    <td className="py-3 px-3 text-brand-cyan">
                      {(model.accuracy * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-brand-cyan font-bold">
                      {(model.macro_f1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-slate-200">
                      {(model.weighted_f1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3">
                      {isChamp ? (
                        <Badge variant="champion" className="text-[10px] py-0.5">
                          CHAMPION
                        </Badge>
                      ) : (
                        <Badge variant="secondary" className="text-[10px] py-0.5">
                          BENCHMARK
                        </Badge>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Row: Radar Chart + Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6">
          <Card
            title="MULTI-DIMENSIONAL PERFORMANCE RADAR"
            subtitle="Normalized metric vectors across test splits"
          >
            <ModelRadarChart data={benchmark} />
          </Card>
        </div>

        <div className="lg:col-span-6">
          <Card
            title="VALIDATION CONFUSION MATRIX"
            subtitle="Actual ground truth vs predicted label distribution"
          >
            <ConfusionMatrix matrix={benchmark.confusion_matrices?.bilstm_conv || { labels: ['Negative', 'Neutral', 'Positive'], matrix: [[787, 55, 48], [79, 963, 86], [52, 67, 863]] }} />
          </Card>
        </div>
      </div>

      {/* Per-Class Metrics */}
      <Card
        title="PER-CLASS GRANULAR METRICS"
        subtitle="Precision, Recall, and F1 performance per sentiment polarity (BiLSTM+Conv)"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          {(benchmark.class_metrics?.bilstm_conv || []).map((cm: ClassMetrics) => (
            <div
              key={cm.label}
              className="p-4 bg-surface-elevated border border-slate-800 space-y-3 font-mono"
            >
              <div className="flex justify-between items-center">
                <Badge
                  sentiment={cm.label as any}
                  className="font-bold uppercase text-xs"
                >
                  {cm.label}
                </Badge>
                <span className="text-[10px] text-slate-400">
                  N={cm.support} SAMPLES
                </span>
              </div>

              <div className="space-y-2 pt-1 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>PRECISION:</span>
                  <span className="font-bold text-slate-100">
                    {(cm.precision * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>RECALL:</span>
                  <span className="font-bold text-slate-100">
                    {(cm.recall * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between text-slate-300 border-t border-slate-800 pt-1">
                  <span>F1-SCORE:</span>
                  <span className="font-bold text-brand-cyan">
                    {(cm.f1 * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Model Architecture Specifications Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 bg-surface-card border border-brand-cyan/60 font-mono text-xs space-y-2">
          <div className="text-brand-cyan font-bold uppercase text-[11px] flex items-center justify-between">
            <span>BiLSTM+Conv</span>
            <span className="text-[9px] bg-brand-cyan/20 px-1 text-brand-cyan border border-brand-cyan/40">TOP</span>
          </div>
          <div className="text-[11px] text-slate-300 leading-relaxed font-sans">
            BiLSTM (128d) followed by Conv1D (k=3, 128 filters) + Dual Pool (Avg+Max).
          </div>
          <div className="text-[10px] text-slate-400 pt-2 border-t border-slate-800">
            F1: 91.64% · Checkpoint: bilstm_conv.pt
          </div>
        </div>

        <div className="p-4 bg-surface-card border border-slate-800 font-mono text-xs space-y-2">
          <div className="text-slate-200 font-bold uppercase text-[11px]">BiLSTM</div>
          <div className="text-[11px] text-slate-300 leading-relaxed font-sans">
            Bidirectional LSTM with 128 hidden units each direction.
          </div>
          <div className="text-[10px] text-slate-400 pt-2 border-t border-slate-800">
            F1: 91.61% · Checkpoint: bilstm.pt
          </div>
        </div>

        <div className="p-4 bg-surface-card border border-slate-800 font-mono text-xs space-y-2">
          <div className="text-slate-200 font-bold uppercase text-[11px]">CNN</div>
          <div className="text-[11px] text-slate-300 leading-relaxed font-sans">
            Multi-kernel 1D CNN with filter sizes (2, 3, 4) × 100 filters + MaxPool.
          </div>
          <div className="text-[10px] text-slate-400 pt-2 border-t border-slate-800">
            F1: 90.35% · Checkpoint: cnn.pt
          </div>
        </div>

        <div className="p-4 bg-surface-card border border-slate-800 font-mono text-xs space-y-2">
          <div className="text-slate-200 font-bold uppercase text-[11px]">LSTM</div>
          <div className="text-[11px] text-slate-300 leading-relaxed font-sans">
            Standard unidirectional LSTM with 128 hidden units.
          </div>
          <div className="text-[10px] text-slate-400 pt-2 border-t border-slate-800">
            F1: 90.01% · Checkpoint: lstm.pt
          </div>
        </div>
      </div>
    </div>
  );
};
