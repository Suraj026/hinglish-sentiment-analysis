import React from 'react';
import { clsx } from 'clsx';
import { Badge } from '../ui/Badge';
import { SentimentResult } from '../../types';

export interface LiveFeedProps {
  items: SentimentResult[];
  maxItems?: number;
  onSelect?: (item: SentimentResult) => void;
}

export const LiveFeed: React.FC<LiveFeedProps> = ({
  items,
  maxItems = 10,
  onSelect,
}) => {
  const displayItems = items.slice(0, maxItems);

  return (
    <div className="divide-y divide-slate-800/80 font-mono text-xs">
      {displayItems.length === 0 ? (
        <div className="py-8 text-center text-slate-500">
          AWAITING STREAM INGESTION...
        </div>
      ) : (
        displayItems.map((item, index) => (
          <div
            key={index}
            onClick={() => onSelect?.(item)}
            className={clsx(
              'py-2.5 px-3 flex items-center justify-between gap-4 transition-colors',
              'hover:bg-surface-elevated/60',
              onSelect && 'cursor-pointer'
            )}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] text-slate-400">
                  {item.model_name.toUpperCase()}
                </span>
                <span className="text-[10px] text-slate-400">·</span>
                <span className="text-[10px] text-slate-400">
                  {item.processing_time_ms}ms
                </span>
                {item.language_mix.is_hinglish && (
                  <span className="text-[9px] px-1 bg-brand-indigo/30 text-brand-indigo border border-brand-indigo/50">
                    HINGLISH
                  </span>
                )}
              </div>
              <p className="text-slate-200 truncate font-sans text-xs">
                {item.text}
              </p>
            </div>

            <div className="flex items-center gap-2 flex-shrink-0">
              <Badge
                sentiment={item.sentiment}
                score={`${Math.round(item.confidence * 100)}%`}
              >
                {item.sentiment}
              </Badge>
            </div>
          </div>
        ))
      )}
    </div>
  );
};
