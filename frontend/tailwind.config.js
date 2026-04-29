/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{svelte,ts}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: { 950: "#0e0e14", 900: "#14141c", 800: "#1a1a22", 700: "#2a2a35", 600: "#404050" },
        accent: { 500: "#6366f1", 600: "#5957d8", 400: "#818cf8" },
        violet2: { 500: "#8b5cf6" },
        amber2: { 500: "#f59e0b" },
      },
      fontFamily: {
        sans: ['ui-sans-serif', 'system-ui', '-apple-system', 'Segoe UI', 'Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
