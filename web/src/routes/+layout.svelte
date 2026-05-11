<script lang="ts">
  let { children } = $props();
</script>

<svelte:head>
  <title>ProteinDesignSpace</title>
</svelte:head>

<main>
  {@render children()}
</main>

<style>
  :global(body) {
    margin: 0;
    background: #0f0f10;
    color: #e6e6e6;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  main {
    max-width: 960px;
    margin: 0 auto;
    padding: 32px 24px;
  }
</style>
