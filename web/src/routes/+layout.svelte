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
  :global(body) {
    margin: 0;
    background: #0b0c10;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
  }
  main {
    max-width: 1280px;
    margin: 0 auto;
    padding: 20px 24px 64px;
  }
</style>
