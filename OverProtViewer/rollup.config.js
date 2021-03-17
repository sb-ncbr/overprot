import resolve from '@rollup/plugin-node-resolve';

export default {
  input: 'lib/OverProtViewer.js',
  output: {
    file: 'dist/overprot-viewer.js',
    format: 'iife',
    globals: {d3: 'd3'}
  },
  external: ['d3'],
  plugins: [resolve()]
};