<script lang="ts">
  let { children } = $props();
</script>

<svelte:head>
  <title>denovo.run — Browser-Native WebGPU Protein Design</title>
</svelte:head>

<main>
  {@render children()}
</main>

<style>
  :global(:root),
  :global(:root[data-theme='dark']) {
    --bg-page: #0b0c10;
    --bg-surface: #13151b;
    --bg-surface-2: #171a23;
    --bg-surface-hover: #1c202c;
    --bg-surface-selected: #151c28;
    --bg-canvas: #0d0e13;
    --bg-input: #0d0e11;
    --bg-pill: #1e293b;
    --border: #262932;
    --border-subtle: #1e222b;
    --border-strong: #334155;
    --text-primary: #f8fafc;
    --text-body: #e2e8f0;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --accent: #3b82f6;
    --accent-bright: #60a5fa;
    --accent-soft: #93c5fd;
    --ok-text: #4ade80;
    --ok-bg: #0f291a;
    --ok-border: #166534;
    --warn-text: #fde68a;
    --warn-bg: #291f0e;
    --warn-border: #b45309;
    --err-text: #fecaca;
    --err-strong: #f87171;
    --err-bg: #2a1215;
    --err-border: #991b1b;
    --node-shadow: 0 6px 18px rgba(0, 0, 0, 0.4);
    --overlay-bg: rgba(13, 14, 17, 0.85);
  }

  :global(:root[data-theme='light']) {
    --bg-page: #f5f7fb;
    --bg-surface: #ffffff;
    --bg-surface-2: #f8fafc;
    --bg-surface-hover: #eef2f7;
    --bg-surface-selected: #eff6ff;
    --bg-canvas: #f1f5f9;
    --bg-input: #ffffff;
    --bg-pill: #e0ecff;
    --border: #d8e0ec;
    --border-subtle: #e6ecf5;
    --border-strong: #cbd5e1;
    --text-primary: #0f172a;
    --text-body: #1e293b;
    --text-secondary: #334155;
    --text-muted: #475569;
    --text-dim: #64748b;
    --accent: #2563eb;
    --accent-bright: #1d4ed8;
    --accent-soft: #1d4ed8;
    --ok-text: #15803d;
    --ok-bg: #dcfce7;
    --ok-border: #86efac;
    --warn-text: #92400e;
    --warn-bg: #fef3c7;
    --warn-border: #f59e0b;
    --err-text: #991b1b;
    --err-strong: #dc2626;
    --err-bg: #fee2e2;
    --err-border: #f87171;
    --node-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
    --overlay-bg: rgba(255, 255, 255, 0.88);
  }

  :global(body) {
    margin: 0;
    background: var(--bg-page);
    color: var(--text-body);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
    transition: background 0.15s ease, color 0.15s ease;
  }
  main {
    max-width: 1600px;
    margin: 0 auto;
    padding: 10px 18px 24px;
  }
</style>
