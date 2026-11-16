import React from 'react';
import { Card } from '../components/ui/Card';
import { MetricTile } from '../components/ui/MetricTile';
import { LiveFeed } from '../components/charts/LiveFeed';
import { SentimentBar } from '../components/charts/SentimentBar';
import { useStats } from '../hooks/useStats';
import { useBenchmark } from '../hooks/useBenchmark';
import { useNavigate } from 'react-router-dom';
import { Activity, Cpu, Database, Zap } from 'lucide-react';
import type { ModelInfo } from '../types';

export const Dashboard: React.FC = () => {
  const { data: stats } = useStats();
  const { data: benchmark } = useBenchmark();
  const navigate = useNavigate();

  return (
    <div className="space-y-6">
      {/* Page Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
        <div>
          <h1 className="font-display text-xl font-bold tracking-tight text-slate-100 uppercase">
            OPERATIONAL OVERVIEW
          </h1>
          <p className="font-mono text-xs text-slate-400 mt-1">
            Real-time inference telemetry & code-mixed NLP intelligence
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => navigate('/analyze')}
            className="h-8 px-4 bg-brand-cyan text-brand-black font-mono font-bold text-xs uppercase tracking-wider hover:bg-brand-cyan-bright transition-colors inline-flex items-center gap-2"
          >
            <Zap className="w-3.5 h-3.5" />
            LIVE INFERENCE CONSOLE
          </button>
        </div>
      </div>

      {/* Row 1: Key Performance Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricTile
          label="TOTAL PROCESSED"
          value={stats?.total_analyzed?.toLocaleString() || '15,847'}
          subValue="STREAMS"
          trend={{ value: '12.4%', positive: true }}
          icon={<Database className="w-4 h-4" />}
        />
        <MetricTile
          label="CHAMPION ACCURACY"
          value="91.6%"
          subValue="BILSTM+CONV"
          highlight={true}
          trend={{ value: '1.7%', positive: true }}
          icon={<Cpu className="w-4 h-4" />}
        />
        <MetricTile
          label="AVG INFERENCE LATENCY"
          value={`${stats?.avg_latency_ms || 12}ms`}
          subValue="LOW JITTER"
          trend={{ value: '0.8ms', positive: true }}
          icon={<Zap className="w-4 h-4" />}
        />
        <MetricTile
          label="CODE-MIXED RATIO"
          value="73%"
          subValue="HINGLISH DENSITY"
          icon={<Activity className="w-4 h-4" />}
        />
      </div>

      {/* Row 2: Live Feed and Model Performance */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Stream Ingestion Feed (2 cols) */}
        <div className="lg:col-span-2">
          <Card
            title="LIVE TELEMETRY STREAM"
            subtitle="Recent code-mixed customer inputs analyzed across pipeline"
            headerRight={
              <span className="flex items-center gap-2 font-mono text-[10px] text-brand-cyan">
                <span className="w-1.5 h-1.5 bg-brand-cyan animate-ping" />
                ACTIVE BUFFER
              </span>
            }
          >
            <LiveFeed
              items={stats?.recent_analyses || []}
              maxItems={6}
              onSelect={() => navigate('/analyze')}
            />
          </Card>
        </div>

        {/* Model Accuracy Comparison */}
        <div>
          <Card
            title="MODEL LEADERBOARD"
            subtitle="Validated on SemEval-2020 Hinglish test set"
            headerRight={
              <button
                onClick={() => navigate('/models')}
                className="font-mono text-[10px] text-brand-cyan hover:underline uppercase"
              >
                DETAILS →
              </button>
            }
          >
            <div className="space-y-4 pt-1">
              {benchmark?.models?.map((model: ModelInfo) => (
                <div key={model.name} className="space-y-1">
                  <div className="flex justify-between items-center text-xs font-mono">
                    <span className="text-slate-300 font-bold">
                      {model.display_name}
                    </span>
                    <span className="text-brand-cyan font-semibold">
                      {Math.round(model.macro_f1 * 1000) / 10}% F1
                    </span>
                  </div>
                  <SentimentBar
                    label=""
                    score={model.accuracy}
                    showPercentage={false}
                  />
                </div>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800/80 font-mono text-[11px] text-slate-400">
              <div className="flex justify-between py-1">
                <span>CHAMPION ARCHITECTURE:</span>
                <span className="text-slate-200">BiLSTM + Conv1D</span>
              </div>
              <div className="flex justify-between py-1">
                <span>INPUT EMBEDDINGS:</span>
                <span className="text-slate-200">128-dim Learned</span>
              </div>
            </div>
          </Card>
        </div>
      </div>

      {/* Row 3: Class Distribution & Language Profile */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card
          title="SENTIMENT CLASS DISTRIBUTION"
          subtitle="Observed sentiment breakdown across recent session inputs"
        >
          <div className="space-y-3 pt-2">
            <SentimentBar
              label="POSITIVE SENTIMENT"
              score={stats?.sentiment_distribution?.positive || 0.42}
              sentiment="positive"
            />
            <SentimentBar
              label="NEUTRAL / AMBIGUOUS"
              score={stats?.sentiment_distribution?.neutral || 0.33}
              sentiment="neutral"
            />
            <SentimentBar
              label="NEGATIVE / CRITICAL"
              score={stats?.sentiment_distribution?.negative || 0.25}
              sentiment="negative"
            />
          </div>
        </Card>

        <Card
          title="CODE-MIXED LINGUISTIC RATIO"
          subtitle="Token language synthesis (English + Devanagari transliteration)"
        >
          <div className="space-y-4 pt-2">
            <div className="space-y-1.5 font-mono text-xs">
              <div className="flex justify-between text-slate-300 text-[11px]">
                <span>HINGLISH (CODE-MIXED)</span>
                <span className="text-brand-indigo font-bold">73%</span>
              </div>
              <div className="h-2 w-full bg-slate-900 border border-slate-800">
                <div className="h-full bg-brand-indigo" style={{ width: '73%' }} />
              </div>
            </div>

            <div className="space-y-1.5 font-mono text-xs">
              <div className="flex justify-between text-slate-300 text-[11px]">
                <span>STANDARD ENGLISH</span>
                <span className="text-brand-cyan font-bold">18%</span>
              </div>
              <div className="h-2 w-full bg-slate-900 border border-slate-800">
                <div className="h-full bg-brand-cyan" style={{ width: '18%' }} />
              </div>
            </div>

            <div className="space-y-1.5 font-mono text-xs">
              <div className="flex justify-between text-slate-300 text-[11px]">
                <span>DEVANAGARI SCRIPT</span>
                <span className="text-slate-400 font-bold">9%</span>
              </div>
              <div className="h-2 w-full bg-slate-900 border border-slate-800">
                <div className="h-full bg-slate-600" style={{ width: '9%' }} />
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
};
