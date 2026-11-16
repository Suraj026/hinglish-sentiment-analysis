import { useQuery } from '@tanstack/react-query';
import { getStats } from '@/lib/api';

export function useStats(refetchInterval = 10_000) {
  return useQuery({
    queryKey: ['stats'],
    queryFn: getStats,
    refetchInterval,
    staleTime: 5_000,
  });
}
