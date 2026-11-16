import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: '#0f131c',
        surface: '#0f131c',
        'surface-card': '#1c1f29',
        'surface-elevated': '#181b25',
        'surface-container': '#1c1f29',
        'surface-container-low': '#181b25',
        'surface-container-high': '#262a34',
        'surface-container-highest': '#31353f',
        primary: '#4cd7f6',
        'primary-container': '#06b6d4',
        secondary: '#c0c1ff',
        'secondary-container': '#3131c0',
        tertiary: '#4edea3',
        'tertiary-container': '#1bbd85',
        'on-surface': '#dfe2ef',
        'on-surface-variant': '#bcc9cd',
        outline: '#869397',
        'outline-variant': '#3d494c',
        error: '#ffb4ab',
        'error-container': '#93000a',
        sidebar: '#0a0e17',

        'brand-obsidian': '#090d16',
        'brand-black': '#001a20',
        'brand-cyan': '#4cd7f6',
        'brand-cyan-bright': '#22d3ee',
        'brand-indigo': '#c0c1ff',
        'brand-emerald': '#4edea3',
        'brand-amber': '#f59e0b',
        'brand-red': '#ef4444',
      },
      fontFamily: {
        headline: ['"Space Grotesk"', 'sans-serif'],
        display: ['"Space Grotesk"', 'sans-serif'],
        body: ['Geist', 'sans-serif'],
        mono: ['"Geist Mono"', 'monospace'],
      },
      borderRadius: {
        DEFAULT: '0',
        none: '0',
        sm: '0',
        md: '0',
        lg: '0',
        xl: '0',
        '2xl': '0',
        '3xl': '0',
        full: '0',
      },
      spacing: {
        'unit': '4px',
        'unit-2': '8px',
      },
      fontSize: {
        'display': ['2rem', { lineHeight: '1.2', fontWeight: '700' }],
        'headline-lg': ['1.5rem', { lineHeight: '1.3', fontWeight: '700' }],
        'headline-md': ['1.25rem', { lineHeight: '1.3', fontWeight: '600' }],
      },
    },
  },
  plugins: [],
}

export default config
