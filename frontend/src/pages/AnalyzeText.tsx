import React, { useState } from 'react';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { SentimentBar } from '../components/charts/SentimentBar';
import { ModelSelector } from '../components/features/ModelSelector';
import { TokenInspector } from '../components/features/TokenInspector';
import { LanguageMixBadge } from '../components/features/LanguageMixBadge';
import { useAnalyze } from '../hooks/useAnalyze';
import { ModelName, SentimentResult } from '../types';
import { Terminal, Clock, Cpu } from 'lucide-react';

const EXAMPLES = [
  'Yeh movie bahut acchi hai 😊',
  'Mujhe bilkul pasand nahi aaya, worst experience',
  'Aaj ka din theek thaak raha, nothing special',
  'Product delivery time par hui par packaging kharab thi',
  'Superb quality! Definitely buying again',
];

export const AnalyzeText: React.FC = () => {
  const [inputText, setInputText] = useState('');
  const [selectedModel, setSelectedModel] = useState<ModelName>('bilstm_conv');
  const { analyze, result, isLoading, history, clearHistory } = useAnalyze();

  const handleAnalyze = () => {
    if (!inputText.trim()) return;
    analyze({ text: inputText, model_name: selectedModel });
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAnalyze();
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
        <div>
          <h1 className="font-display text-xl font-bold tracking-tight text-slate-100 uppercase">
            LIVE SENTIMENT TELEMETRY
          </h1>
          <p className="font-mono text-xs text-slate-400 mt-1">
            Real-time inference on code-mixed Hinglish strings with per-token attribution
          </p>
        </div>

        <div className="flex items-center gap-3">
          <ModelSelector
            value={selectedModel}
            onChange={setSelectedModel}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input Stream & Token Inspector (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <Card
            title="INGESTION BUFFER"
            subtitle="Enter Hinglish, Romanized Hindi, or English text for neural analysis"
            headerRight={
              <span className="font-mono text-[10px] text-slate-400">
                CTRL + ENTER TO SUBMIT
              </span>
            }
          >
            <div className="space-y-4">
              <div className="relative">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="> Type or paste code-mixed review text here... (e.g. 'Yeh product bohot hi bekar nikla')"
                  className="w-full h-36 bg-[#0a0e17] border border-[#3d494c] p-3 font-mono text-xs text-white placeholder-slate-500 focus:outline-none focus:border-brand-cyan resize-none"
                  style={{ color: '#ffffff' }}
                />
                <div className="absolute right-3 bottom-3 font-mono text-[10px] text-slate-400">
                  {inputText.length} CHARS
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <Button
                    onClick={handleAnalyze}
                    isLoading={isLoading}
                    disabled={!inputText.trim()}
                  >
                    RUN INFERENCE
                  </Button>
                  <Button
                    variant="secondary"
                    onClick={() => setInputText('')}
                    disabled={!inputText}
                  >
                    CLEAR
                  </Button>
                </div>

                <span className="font-mono text-[11px] text-slate-400">
                  SELECTED: <span className="text-brand-cyan font-bold uppercase">{selectedModel}</span>
                </span>
              </div>
            </div>
          </Card>

          {/* Preset Prompts / Examples */}
          <Card title="SAMPLE EVALUATION STRINGS" subtitle="Click to load pre-annotated test cases">
            <div className="flex flex-wrap gap-2">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  onClick={() => setInputText(ex)}
                  className="px-2.5 py-1.5 bg-surface-elevated border border-slate-800 text-xs font-mono text-slate-300 hover:border-brand-cyan/60 hover:text-slate-100 transition-colors text-left truncate max-w-full"
                >
                  {ex}
                </button>
              ))}
            </div>
          </Card>

          {/* Token Inspector */}
          {result && (
            <Card
              title="TOKEN-LEVEL DECOMPOSITION"
              subtitle="Lexical normalization and code-mixed token stream"
              headerRight={
                <span className="font-mono text-[10px] text-slate-400">
                  {result.tokens?.length || 0} TOKENS
                </span>
              }
            >
              <TokenInspector tokens={result.tokens} />
            </Card>
          )}
        </div>

        {/* Right Column: Classification Results & Telemetry (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          <Card
            title="INFERENCE OUTPUT"
            subtitle="Classification decision & probability distribution"
            highlight={Boolean(result)}
          >
            {result ? (
              <div className="space-y-6">
                {/* Sentiment Verdict Banner */}
                <div className="p-4 bg-surface-elevated border border-slate-800 text-center space-y-2">
                  <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400">
                    PREDICTED SENTIMENT
                  </span>
                  <div className="flex items-center justify-center gap-3">
                    <Badge
                      sentiment={result.sentiment}
                      className="text-base px-3 py-1 font-bold"
                    >
                      {result.sentiment}
                    </Badge>
                  </div>
                  <div className="font-mono text-2xl font-bold text-slate-100">
                    {Math.round(result.confidence * 1000) / 10}%
                  </div>
                  <div className="font-mono text-[10px] text-slate-400">
                    MODEL CONFIDENCE SCORE
                  </div>
                </div>

                {/* Probability Distribution Bars */}
                <div className="space-y-3">
                  <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block">
                    CLASS PROBABILITY DISTRIBUTION
                  </span>
                  <SentimentBar
                    label="POSITIVE"
                    score={result.scores.positive}
                    sentiment="positive"
                  />
                  <SentimentBar
                    label="NEUTRAL"
                    score={result.scores.neutral}
                    sentiment="neutral"
                  />
                  <SentimentBar
                    label="NEGATIVE"
                    score={result.scores.negative}
                    sentiment="negative"
                  />
                </div>

                {/* Linguistic Mix Tag */}
                <div className="pt-3 border-t border-slate-800">
                  <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-2">
                    LANGUAGE COMPOSITION
                  </span>
                  <LanguageMixBadge languageMix={result.language_mix} />
                </div>

                {/* Telemetry metadata */}
                <div className="pt-3 border-t border-slate-800 grid grid-cols-2 gap-3 font-mono text-xs">
                  <div className="flex items-center gap-2 text-slate-400">
                    <Clock className="w-3.5 h-3.5 text-brand-cyan" />
                    <span>LATENCY:</span>
                    <span className="text-slate-200 font-bold">
                      {result.processing_time_ms}ms
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-slate-400">
                    <Cpu className="w-3.5 h-3.5 text-brand-cyan" />
                    <span>ENGINE:</span>
                    <span className="text-slate-200 font-bold uppercase truncate">
                      {result.model_name}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="py-16 text-center space-y-3">
                <Terminal className="w-8 h-8 mx-auto text-slate-600" />
                <p className="font-mono text-xs text-slate-400">
                  SUBMIT TEXT TO GENERATE SENTIMENT TELEMETRY
                </p>
              </div>
            )}
          </Card>

          {/* Session History */}
          {history.length > 0 && (
            <Card
              title="SESSION TELEMETRY LOG"
              subtitle="Previous inferences evaluated in this session"
              headerRight={
                <button
                  onClick={clearHistory}
                  className="font-mono text-[10px] text-slate-400 hover:text-slate-200 uppercase"
                >
                  CLEAR
                </button>
              }
            >
              <div className="divide-y divide-slate-800 font-mono text-xs max-h-60 overflow-y-auto">
                {history.map((h: SentimentResult, i: number) => (
                  <div key={i} className="py-2 flex items-center justify-between gap-3">
                    <div className="truncate flex-1">
                      <p className="text-slate-300 truncate font-sans text-xs">
                        {h.text}
                      </p>
                      <span className="text-[10px] text-slate-400">
                        {h.model_name.toUpperCase()} · {h.processing_time_ms}ms
                      </span>
                    </div>
                    <Badge sentiment={h.sentiment}>
                      {h.sentiment.slice(0, 3)}
                    </Badge>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
