import React from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { BenchmarkData, ModelInfo } from '../../types';

export interface ModelRadarChartProps {
  data: BenchmarkData;
}

export const ModelRadarChart: React.FC<ModelRadarChartProps> = ({ data }) => {
  const radarMetrics = [
    { metric: 'Accuracy', ...Object.fromEntries(data.models.map((m: ModelInfo) => [m.name, Math.round(m.accuracy * 100)])) },
    { metric: 'Macro F1', ...Object.fromEntries(data.models.map((m: ModelInfo) => [m.name, Math.round(m.macro_f1 * 100)])) },
    { metric: 'Weighted F1', ...Object.fromEntries(data.models.map((m: ModelInfo) => [m.name, Math.round(m.weighted_f1 * 100)])) },
  ];

  const colors: Record<string, string> = {
    bilstm_conv: '#4cd7f6', // Cyan (champion)
    bilstm: '#c0c1ff',      // Indigo
    cnn: '#4edea3',         // Emerald
    lstm: '#f59e0b',        // Amber
  };

  return (
    <div className="w-full h-64 font-mono text-xs">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={radarMetrics} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis dataKey="metric" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <PolarRadiusAxis domain={[80, 95]} stroke="#334155" tick={{ fill: '#64748b', fontSize: 10 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f172a',
              borderColor: '#334155',
              borderRadius: 0,
              fontSize: '11px',
              fontFamily: 'monospace',
            }}
          />
          <Legend
            wrapperStyle={{
              fontSize: '11px',
              fontFamily: 'monospace',
              paddingTop: '8px',
            }}
          />
          {data.models.map((m: ModelInfo) => (
            <Radar
              key={m.name}
              name={m.display_name}
              dataKey={m.name}
              stroke={colors[m.name] || '#94a3b8'}
              fill={colors[m.name] || '#94a3b8'}
              fillOpacity={m.name === 'bilstm_conv' ? 0.35 : 0.1}
            />
          ))}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
};
