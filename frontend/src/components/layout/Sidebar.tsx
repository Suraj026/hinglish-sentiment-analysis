import { NavLink } from 'react-router-dom';
import { LayoutDashboard, FlaskConical, ScanText, Activity } from 'lucide-react';
import { cn } from '@/lib/utils';

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', Icon: LayoutDashboard },
  { to: '/analyze', label: 'Analyze Text', Icon: ScanText },
  { to: '/models', label: 'Model Lab', Icon: FlaskConical },
];

export function Sidebar() {
  return (
    <aside
      className="flex flex-col shrink-0"
      style={{
        width: '240px',
        background: '#0a0e17',
        borderRight: '1px solid #3d494c',
        height: '100vh',
        position: 'sticky',
        top: 0,
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-3 px-5 py-4"
        style={{ borderBottom: '1px solid #3d494c' }}
      >
        <Activity size={20} style={{ color: '#4cd7f6' }} />
        <span
          className="font-headline text-base font-bold tracking-wide"
          style={{ color: '#4cd7f6' }}
        >
          HinglishPulse
        </span>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-1 p-3 flex-1">
        {NAV_ITEMS.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2 text-sm transition-colors',
                'font-body',
                isActive
                  ? 'bg-surface-container-high text-primary border-l-2 border-primary'
                  : 'text-on-surface-variant hover:bg-surface-container hover:text-on-surface'
              )
            }
            style={{ borderRadius: 0 }}
          >
            {({ isActive }) => (
              <>
                <Icon
                  size={16}
                  style={{ color: isActive ? '#4cd7f6' : undefined, flexShrink: 0 }}
                />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div
        className="px-5 py-3 text-xs font-mono"
        style={{ borderTop: '1px solid #3d494c', color: '#869397' }}
      >
        v0.1.0 · HinglishPulse
      </div>
    </aside>
  );
}
