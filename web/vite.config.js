import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: {
    exclude: ['zprj_loader.js'],
  },
});
