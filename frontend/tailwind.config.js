/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          primary: '#059669', // Emerald 600
          secondary: '#10B981', // Emerald 500
          accent: '#6366F1', // Indigo 500
          slate: {
            900: '#0f172a',
            800: '#1e293b',
            700: '#334155',
          },
        },
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeOutDown: {
          '0%': { opacity: '1', transform: 'translateY(0)' },
          '100%': { opacity: '0', transform: 'translateY(-10px)' },
        },
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.3s ease-out',
        'fade-out-down': 'fadeOutDown 0.3s ease-in',
      },
    },
  },
  plugins: [],
};
