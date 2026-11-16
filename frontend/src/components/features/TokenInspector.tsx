import React from 'react';

// ─── Heuristic: assign token sentiment based on known word lists ──────────────
const POSITIVE_WORDS = new Set([
  'acchi', 'achi', 'bahut', 'accha', 'happy', 'great', 'best', 'love', 'good',
  'amazing', 'wonderful', 'soooo', 'excellent', 'mast', 'zabardast', 'shaandaar',
]);
const NEGATIVE_WORDS = new Set([
  'nahi', 'nhi', 'pasand', 'bilkul', 'bura', 'buri', 'bad', 'worst', 'hate',
  'samajh', 'problem', 'issue', 'kharab', 'bekar', 'faltu',
]);

type TokenSentiment = 'positive' | 'negative' | 'neutral';

function classifyToken(token: string): TokenSentiment {
  const t = token.toLowerCase();
  if (POSITIVE_WORDS.has(t)) return 'positive';
  if (NEGATIVE_WORDS.has(t)) return 'negative';
  return 'neutral';
}

const SENTIMENT_STYLES: Record<TokenSentiment, React.CSSProperties> = {
  positive: {
    background: 'rgba(78,222,163,0.12)',
    border: '1px solid rgba(78,222,163,0.4)',
    color: '#4edea3',
    textDecorationLine: 'underline',
    textDecorationColor: 'rgba(78,222,163,0.6)',
    textDecorationStyle: 'solid',
    textUnderlineOffset: '2px',
  },
  negative: {
    background: 'rgba(239,68,68,0.12)',
    border: '1px solid rgba(239,68,68,0.4)',
    color: '#ef4444',
  },
  neutral: {
    background: '#1c1f29',
    border: '1px solid #3d494c',
    color: '#bcc9cd',
  },
};

const SCORE_BY_SENTIMENT: Record<TokenSentiment, string> = {
  positive: '+0.82',
  negative: '−0.74',
  neutral: '±0.00',
};

interface TokenInspectorProps {
  tokens: string[];
}

export const TokenInspector: React.FC<TokenInspectorProps> = ({ tokens }) => {
  if (!tokens || tokens.length === 0) return null;

  return (
    <div style={{ marginTop: '24px' }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          marginBottom: '12px',
        }}
      >
        <span
          style={{
            fontFamily: 'Geist Mono, monospace',
            fontSize: '11px',
            fontWeight: 500,
            letterSpacing: '0.12em',
            color: '#4cd7f6',
            textTransform: 'uppercase',
          }}
        >
          TOKEN STREAM ANALYSIS
        </span>
        <span
          style={{
            background: '#1c1f29',
            border: '1px solid #3d494c',
            padding: '1px 6px',
            fontFamily: 'Geist Mono, monospace',
            fontSize: '10px',
            color: '#869397',
          }}
        >
          {tokens.length} tokens
        </span>
      </div>

      {/* Token chips */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '6px',
        }}
      >
        {tokens.map((token, idx) => {
          const sentiment = classifyToken(token);
          const style = SENTIMENT_STYLES[sentiment];
          const score = SCORE_BY_SENTIMENT[sentiment];

          return (
            <div
              key={idx}
              title={`Token #${idx} · score ${score}`}
              style={{
                position: 'relative',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '3px 8px',
                borderRadius: 0,
                fontFamily: 'Geist Mono, monospace',
                fontSize: '12px',
                cursor: 'default',
                userSelect: 'none',
                transition: 'opacity 0.15s',
                ...style,
              }}
            >
              <span
                style={{
                  fontSize: '9px',
                  color: 'inherit',
                  opacity: 0.6,
                  marginRight: '2px',
                }}
              >
                [{idx}]
              </span>
              {token}
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          gap: '16px',
          marginTop: '12px',
        }}
      >
        {(['positive', 'negative', 'neutral'] as TokenSentiment[]).map((s) => (
          <div
            key={s}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              fontFamily: 'Geist Mono, monospace',
              fontSize: '10px',
              color: '#869397',
              letterSpacing: '0.06em',
            }}
          >
            <div
              style={{
                width: '10px',
                height: '10px',
                ...SENTIMENT_STYLES[s],
                display: 'inline-block',
              }}
            />
            {s.toUpperCase()}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TokenInspector;
