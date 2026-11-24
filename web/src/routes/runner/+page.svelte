<script lang="ts">
  // Hidden host-GPU execution harness loaded inside headless Chromium by server/selfhost-server.mjs.
  // Reuses ExecutionScheduler's exact 22 WebGPU workers + single-slot VRAM lifecycle + OPFS weight cache.
  import { onMount } from 'svelte';
  import { ExecutionScheduler } from '$lib/engine/scheduler';
  import { setCachePluginRegistry } from '$lib/engine/opfs-cache';
  import { defaultPluginRegistry } from '$lib/plugins';
  import { decodeWireValue, encodeWireValue } from '$lib/engine/remote-codec';
  import { formatWebGpuAdapterSummary, requestResilientWebGpuAdapter } from '$lib/engine/webgpu-device';
  import type { CanonicalArtifact, NodeProgressEvent, WorkflowNode } from '$lib/engine/types';

  interface HostRunnerApi {
    getHealth: () => {
      ready: boolean;
      gpu: { vendor: string; architecture: string; description: string; fallback: boolean } | null;
      activeAdapter: string | null;
      error: string | null;
    };
    runNode: (
      jobId: string,
      payload: { node: WorkflowNode; inputs: unknown; maxFoldResidues?: number },
    ) => Promise<{ outputs: unknown; elapsedMs: number }>;
    cancel: (jobId: string) => Promise<void>;
    releaseGpu: () => Promise<void>;
  }

  interface HostWindow extends Window {
    __denovoHostRunner?: HostRunnerApi;
    __denovoHostEmit?: (event: Record<string, unknown>) => void;
  }

  let statusText = $state('Initializing denovo.run host WebGPU runner...');
  let gpuText = $state('Probing GPU...');

  onMount(() => {
    const win = window as unknown as HostWindow;
    setCachePluginRegistry(defaultPluginRegistry);
    const scheduler = new ExecutionScheduler({
      gpuIdleReleaseMs: 120_000,
      pluginRegistry: defaultPluginRegistry,
    });
    let gpuInfo: { vendor: string; architecture: string; description: string; fallback: boolean } | null = null;
    let initError: string | null = null;
    let ready = false;
    const activeAborts = new Map<string, AbortController>();

    const emit = (ev: Record<string, unknown>) => {
      try {
        win.__denovoHostEmit?.(ev);
      } catch {
        /* ignore bridge errors */
      }
    };

    win.__denovoHostRunner = {
      getHealth: () => ({
        ready,
        gpu: gpuInfo,
        activeAdapter: scheduler.getActiveGpuAdapter(),
        error: initError,
      }),
      runNode: async (jobId: string, payload: { node: WorkflowNode; inputs: unknown; maxFoldResidues?: number }) => {
        const ctrl = new AbortController();
        activeAborts.set(jobId, ctrl);
        try {
          const decodedInputs = decodeWireValue<Record<string, CanonicalArtifact>>(payload.inputs);
          const res = await scheduler.runSingleGpuNode(
            payload.node,
            decodedInputs,
            (ev: NodeProgressEvent) => {
              emit({
                type: 'progress',
                jobId,
                status: ev.status,
                phase: ev.phase ?? 'running',
                progress: ev.progress,
                detail: ev.detail ?? ev.status,
              });
            },
            ctrl.signal,
            payload.maxFoldResidues,
          );
          return {
            outputs: encodeWireValue(res.outputs),
            elapsedMs: res.elapsedMs,
          };
        } finally {
          activeAborts.delete(jobId);
        }
      },
      cancel: async (jobId: string) => {
        const ctrl = activeAborts.get(jobId);
        if (!ctrl) return;
        ctrl.abort();
        await scheduler.releaseActiveGpu();
      },
      releaseGpu: async () => {
        await scheduler.releaseActiveGpu();
      },
    };

    void (async () => {
      try {
        const probe = await requestResilientWebGpuAdapter({
          retries: 3,
          retryDelayMs: 250,
          allowSoftwareFallback: true,
        });
        const summary = formatWebGpuAdapterSummary(probe.info, probe.hasF16);
        gpuInfo = {
          vendor: probe.info.vendor || 'unknown',
          architecture: probe.info.architecture || '',
          description: probe.info.description || probe.info.device || summary || 'WebGPU GPU',
          fallback: probe.isSoftwareFallback,
        };
        gpuText = `${gpuInfo.vendor} ${gpuInfo.description}${gpuInfo.fallback ? ' (software)' : ''}`;
        statusText = 'Host WebGPU runner ready.';
      } catch (err) {
        initError = err instanceof Error ? err.message : String(err);
        statusText = `WebGPU probe error: ${initError}`;
      } finally {
        ready = true;
        emit({ type: 'runner-ready', gpu: gpuInfo, error: initError });
      }
    })();

    return () => {
      void scheduler.destroy();
    };
  });
</script>

<main style="font-family: ui-monospace, monospace; padding: 16px; color: #cbd5e1; background: #0b0c10; min-height: 100vh;">
  <h1 style="font-size: 14px; margin: 0 0 8px;">denovo.run · Self-Hosted WebGPU Runner</h1>
  <div style="font-size: 12px;">Status: {statusText}</div>
  <div style="font-size: 12px; margin-top: 4px;">GPU: {gpuText}</div>
</main>
