import { describe, expect, it } from 'vitest';
import { decodeWireValue, encodeWireValue } from '$lib/engine/remote-codec';
import {
  defaultComputeBackendConfig,
  estimateNodeActiveWeightsMB,
  runNodeOnRemoteServer,
  shouldRouteNodeToRemote,
  type ComputeBackendConfig,
} from '$lib/engine/compute-backend';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import type { CanonicalArtifact, StructureArtifact, WorkflowNode } from '$lib/engine/types';

describe('self-hosted remote WebGPU backend', () => {
  it('round-trips TypedArrays (Float32Array, Uint8Array) and special numbers cleanly over JSON wire', () => {
    const struct: StructureArtifact = {
      kind: 'StructureArtifact',
      id: 'struct-test',
      label: 'Test Complex',
      sourceModelId: 'pdb-input',
      length: 2,
      sequence: 'AC',
      caCoords: new Float32Array([1.25, -2.5, 3.75, 4.0, 5.125, -6.25]),
      backbone4Coords: new Float32Array(24).fill(0.5),
      observedBackboneMask: new Uint8Array([0x0f, 0x07]),
      pdbText: 'ATOM      1  CA  ALA A   1       1.250  -2.500   3.750  1.00 20.00           C  \nEND\n',
      geometry: computeGeometryStats(new Float32Array([1.25, -2.5, 3.75, 4.0, 5.125, -6.25])),
    };
    const payload = {
      structure: struct,
      nanVal: Number.NaN,
      infVal: Number.POSITIVE_INFINITY,
    };

    const jsonStr = JSON.stringify(encodeWireValue(payload));
    const decoded = decodeWireValue<typeof payload>(JSON.parse(jsonStr));

    expect(decoded.structure.caCoords).toBeInstanceOf(Float32Array);
    expect(Array.from(decoded.structure.caCoords)).toEqual([1.25, -2.5, 3.75, 4.0, 5.125, -6.25]);
    expect(decoded.structure.observedBackboneMask).toBeInstanceOf(Uint8Array);
    expect(Array.from(decoded.structure.observedBackboneMask!)).toEqual([0x0f, 0x07]);
    expect(decoded.structure.pdbText).toContain('ALA A   1');
    expect(Number.isNaN(decoded.nanVal)).toBe(true);
    expect(decoded.infVal).toBe(Number.POSITIVE_INFINITY);
  });

  it('routes nodes accurately across local, remote, and hybrid modes', () => {
    const pdbNode: WorkflowNode = { id: 'p1', adapterId: 'pdb-input', params: {} };
    const gninaNode: WorkflowNode = {
      id: 'g1',
      adapterId: 'gnina',
      params: { precision: 'fp16', taskMode: 'dock' },
    };
    const af3Node: WorkflowNode = {
      id: 'f1',
      adapterId: 'localfold-af2',
      params: { model: 'af3' },
    };

    expect(estimateNodeActiveWeightsMB(gninaNode)).toBe(4);
    expect(estimateNodeActiveWeightsMB(af3Node)).toBeGreaterThan(200);

    const localCfg = defaultComputeBackendConfig();
    expect(shouldRouteNodeToRemote(af3Node, localCfg, { localWebGpuAvailable: true })).toBe(false);

    const remoteCfg: ComputeBackendConfig = { ...localCfg, mode: 'remote', serverUrl: 'http://host:8787' };
    expect(shouldRouteNodeToRemote(pdbNode, remoteCfg, { localWebGpuAvailable: true })).toBe(false); // CPU node stays local
    expect(shouldRouteNodeToRemote(gninaNode, remoteCfg, { localWebGpuAvailable: true })).toBe(true);
    expect(shouldRouteNodeToRemote(af3Node, remoteCfg, { localWebGpuAvailable: true })).toBe(true);

    const hybridCfg: ComputeBackendConfig = { ...localCfg, mode: 'hybrid', hybridThresholdMB: 40 };
    // With local hardware WebGPU: light GNINA (4 MB) runs locally, heavy AF3 (>200 MB) runs remotely
    expect(shouldRouteNodeToRemote(gninaNode, hybridCfg, { localWebGpuAvailable: true })).toBe(false);
    expect(shouldRouteNodeToRemote(af3Node, hybridCfg, { localWebGpuAvailable: true })).toBe(true);
    // Without local hardware WebGPU (e.g. phone/iPad): even GNINA routes to host GPU
    expect(shouldRouteNodeToRemote(gninaNode, hybridCfg, { localWebGpuAvailable: false })).toBe(true);
  });

  it('streams SSE progress and decodes TypedArray results from remote host GPU', async () => {
    const encodedOut = encodeWireValue({
      structure: {
        kind: 'StructureArtifact',
        id: 'remote-out',
        label: 'Docked',
        sourceModelId: 'gnina',
        length: 1,
        sequence: 'A',
        caCoords: new Float32Array([1, 2, 3]),
        backbone4Coords: new Float32Array(12),
        pdbText: 'END\n',
        geometry: computeGeometryStats(new Float32Array([1, 2, 3])),
      },
    });

    const sseBody = [
      `event: progress\ndata: ${JSON.stringify({ status: 'running', phase: 'dock', progress: 0.6, detail: 'MC+BFGS search' })}\n\n`,
      `event: result\ndata: ${JSON.stringify({ outputs: encodedOut, elapsedMs: 1234 })}\n\n`,
    ].join('');

    const mockFetch: typeof fetch = async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(sseBody));
            controller.close();
          },
        }),
        { status: 200, headers: { 'Content-Type': 'text/event-stream' } },
      );

    const progressEvents: string[] = [];
    const res = await runNodeOnRemoteServer({
      serverUrl: 'http://127.0.0.1:8787',
      apiToken: 'secret',
      node: { id: 'gnina_1', adapterId: 'gnina', label: 'GNINA 1.3', params: { taskMode: 'dock' } },
      inputs: {} as Record<string, CanonicalArtifact>,
      onProgress: (ev) => {
        if (ev.detail) progressEvents.push(ev.detail);
      },
      fetchImpl: mockFetch,
    });

    expect(progressEvents.some((d) => d.includes('[Host GPU] MC+BFGS search'))).toBe(true);
    const outStruct = res.outputs.structure as StructureArtifact;
    expect(outStruct.caCoords).toBeInstanceOf(Float32Array);
    expect(Array.from(outStruct.caCoords)).toEqual([1, 2, 3]);
    expect(res.elapsedMs).toBe(1234);
  });
});
