import { useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { getHealth } from '@/lib/api';
import { Wifi, WifiOff } from 'lucide-react';

const PAGE_TITLES: Record<string, string> = {
  '/': 'Overview Dashboard',
  '/analyze': 'Analyze Text',
  '/models': 'Model Lab & Benchmarks',
};

export function TopBar() {
  const { pathname } = useLocation();
  const title = PAGE_TITLES[pathname] ?? 'HinglishPulse';

  const { data: health, isError } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 15_000,
    retry: 1,
  });

  const connected = health?.status === 'ok';

  return (
    <header
      className="flex items-center justify-between px-6 h-12 shrink-0 font-body"
      style={{
        background: '#0f131c',
        borderBottom: '1px solid #3d494c',
      }}
    >
      <span className="font-headline font-semibold text-sm" style={{ color: '#dfe2ef' }}>
        {title}
      </span>

      <div className="flex items-center gap-4">
        {/* Backend status */}
        <div className="flex items-center gap-2 text-xs font-mono" style={{ color: '#bcc9cd' }}>
          {!isError && connected ? (
            <>
              <Wifi size={12} style={{ color: '#4edea3' }} />
              <span style={{ color: '#4edea3' }}>API OK</span>
              {health && (
                <span style={{ color: '#869397' }}>· {health.model_count} models</span>
              )}
            </>
          ) : (
            <>
              <WifiOff size={12} style={{ color: '#ffb4ab' }} />
              <span style={{ color: '#ffb4ab' }}>API offline</span>
            </>
          )}
        </div>

        {/* Timestamp */}
        <span className="text-xs font-mono" style={{ color: '#3d494c' }}>
          {new Date().toLocaleTimeString()}
        </span>
      </div>
    </header>
  );
}
