// Lazy Mol* loader. We pull the prebuilt viewer bundle from jsdelivr the first
// time a viewer is created; subsequent components share the same global.
// This matches the pattern used in legacy-model-flow-map/examples/14_*.py.

declare global {
  interface Window {
    // Mol* attaches its public surface here.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    molstar?: any;
  }
}

const MOLSTAR_VERSION = '4';
const CSS_URL = `https://cdn.jsdelivr.net/npm/molstar@${MOLSTAR_VERSION}/build/viewer/molstar.css`;
const JS_URL = `https://cdn.jsdelivr.net/npm/molstar@${MOLSTAR_VERSION}/build/viewer/molstar.js`;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loadPromise: Promise<any> | null = null;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadMolstar(): Promise<any> {
  if (typeof window === 'undefined') {
    return Promise.reject(new Error('Mol* requires a browser'));
  }
  if (window.molstar) return Promise.resolve(window.molstar);
  if (loadPromise) return loadPromise;

  loadPromise = new Promise((resolve, reject) => {
    if (!document.querySelector(`link[href="${CSS_URL}"]`)) {
      const css = document.createElement('link');
      css.rel = 'stylesheet';
      css.href = CSS_URL;
      document.head.appendChild(css);
    }

    const existing = document.querySelector(`script[src="${JS_URL}"]`);
    if (existing) {
      existing.addEventListener('load', () => {
        if (window.molstar) resolve(window.molstar);
        else reject(new Error('Mol* script loaded but no global namespace'));
      });
      return;
    }

    const script = document.createElement('script');
    script.src = JS_URL;
    script.async = true;
    script.onload = () => {
      if (window.molstar) resolve(window.molstar);
      else reject(new Error('Mol* script loaded but no global namespace'));
    };
    script.onerror = () => reject(new Error('Failed to load Mol* from CDN'));
    document.head.appendChild(script);
  });
  return loadPromise;
}
