// Disable SSR everywhere — the playground depends on browser APIs (WebGPU,
// Web Workers). Prerender shells so the static adapter has something to emit.
export const ssr = false;
export const prerender = true;
