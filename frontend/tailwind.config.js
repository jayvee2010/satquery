/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          blue: '#388bfd',
          darkBlue: '#0c2d5a',
          cyan: '#58a6ff',
          green: '#56d364',
          orange: '#da760b',
          red: '#f85149',
          bg: '#0d0f12',
          card: '#161b22',
          border: '#30363d'
        }
      }
    },
  },
  plugins: [],
}
