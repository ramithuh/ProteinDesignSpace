export async function requestHardwareAdapter({ allowSoftwareFallback = true } = {}) {
  if (!globalThis.navigator?.gpu)
    throw new Error("Hardware WebGPU unavailable");
  // Chromium on Linux/Vulkan can return null during the first native probe.
  // Retry high-performance briefly, then fall back to default adapter selection.
  for (let attempt = 0; attempt < 3; attempt++) {
    const adapter =
      (await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      })) ?? (await navigator.gpu.requestAdapter());
    if (adapter) {
      const info = adapter.info ?? {};
      const isSoftware =
        adapter.isFallbackAdapter ||
        info.isFallbackAdapter ||
        /swiftshader|llvmpipe|lavapipe|software|\bwarp\b|basic render/i.test(
          [info.vendor, info.device, info.description].join(" "),
        );
      if (isSoftware && !allowSoftwareFallback)
        throw new Error("Software WebGPU adapters are unsupported");
      return adapter;
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(
    "WebGPU adapter unavailable (three adapter probes failed)",
  );
}
