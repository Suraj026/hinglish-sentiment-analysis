import React from 'react';
import { clsx } from 'clsx';
import { ConfusionMatrixData } from '../../types';

export interface ConfusionMatrixProps {
  matrix: ConfusionMatrixData;
  title?: string;
}

export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  matrix,
  title = 'CONFUSION MATRIX (BEST MODEL: BiLSTM+Conv)',
}) => {
  const { labels, matrix: data } = matrix;
  const flatData = data.flat();
  const totalSamples = flatData.reduce((acc: number, val: number) => acc + val, 0);
  const maxVal = Math.max(...flatData);

  const getIntensity = (val: number) => {
    if (val === 0) return 'bg-slate-900/60 text-slate-600';
    const ratio = val / maxVal;
    if (ratio > 0.7) return 'bg-brand-cyan text-brand-black font-bold';
    if (ratio > 0.4) return 'bg-brand-cyan/60 text-brand-black font-semibold';
    if (ratio > 0.15) return 'bg-brand-cyan/20 text-brand-cyan';
    return 'bg-slate-800 text-slate-400';
  };

  return (
    <div className="space-y-3 font-mono text-xs">
      <div className="flex justify-between items-center text-slate-400 text-[11px]">
        <span className="uppercase tracking-wider">{title}</span>
        <span>TOTAL SAMPLES: {totalSamples}</span>
      </div>

      <div className="overflow-x-auto">
        <div className="min-w-[280px]">
          {/* Column labels (Predicted) */}
          <div className="grid grid-cols-4 gap-1 text-center text-[10px] uppercase tracking-wider text-slate-400 mb-1">
            <div className="text-left pl-2">ACTUAL \ PRED</div>
            {labels.map((lbl: string) => (
              <div key={lbl} className="font-semibold text-slate-300">
                {lbl.slice(0, 3)}
              </div>
            ))}
          </div>

          {/* Matrix rows */}
          <div className="space-y-1">
            {labels.map((rowLabel: string, rIdx: number) => (
              <div key={rowLabel} className="grid grid-cols-4 gap-1 items-center">
                <div className="text-left pl-2 font-semibold text-slate-300 uppercase text-[10px]">
                  {rowLabel.slice(0, 3)}
                </div>
                {data[rIdx].map((val: number, cIdx: number) => (
                  <div
                    key={cIdx}
                    className={clsx(
                      'h-10 flex items-center justify-center text-xs border border-slate-800 transition-colors',
                      getIntensity(val)
                    )}
                    title={`Actual: ${rowLabel}, Predicted: ${labels[cIdx]}: ${val}`}
                  >
                    {val}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-2 border-t border-slate-800/80 text-[10px] text-slate-400">
        <span>Rows: Ground Truth</span>
        <span>Columns: Predictions</span>
      </div>
    </div>
  );
};
