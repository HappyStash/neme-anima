/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{svelte,ts}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: { 950: "#0e0e14", 900: "#14141c", 800: "#1a1a22", 700: "#2a2a35", 600: "#404050" },
        // Standard Tailwind indigo scale around our base 500/600/400 anchors.
        // Several components reference 200/300/700 (PipelineRunner status
        // pills, the training-tab toggle chips); leaving those undefined
        // makes Tailwind drop the class silently — which is exactly what
        // made the "active" state of the resolution-bucket / preset toggles
        // visually indistinguishable from inactive (no bg = parent bg-ink-900
        // showing through, so active and inactive looked the same).
        accent: {
          200: "#c7d2fe",
          300: "#a5b4fc",
          400: "#818cf8",
          500: "#6366f1",
          600: "#5957d8",
          700: "#4338ca",
        },
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
