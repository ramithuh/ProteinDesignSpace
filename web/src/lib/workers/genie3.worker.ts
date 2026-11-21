/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/genie3-webgpu (Apache-2.0 port of aqlaboratory/genie3)
// Generates de novo protein backbones (ideal N/CA/C/O + Cα) and optional 1-step native sequences
// via SE(3)-equivariant DDIM diffusion on WebGPU (~45 MB FP16 weights).

import { parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import { getBenchmarkById } from '$lib/engine/benchmarks';
import {
  buildCanonicalConditionIntent,
  buildDefaultGenie3ConditionFromTemplate,
  compileConditionToGenie3Spec,
  effectiveGenie3DdimSteps,
  resolveGenie3PolyAlaSequence,
  type Genie3CompiledSpec,
} from '$lib/engine/conditioning';
import type { ConditionSpecArtifact, SequenceSet, StructureArtifact } from '$lib/engine/types';

declare const self: DedicatedWorkerGlobalScope;

function fnv1a(str: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

function collectOwnedTransferables(views: Array<ArrayBufferView | undefined>): Transferable[] {
  const seen = new Set<ArrayBuffer>();
  const out: Transferable[] = [];
  for (const v of views) {
    if (!v) continue;
    const buf = v.buffer;
    if (!(buf instanceof ArrayBuffer)) continue;
    if (v.byteOffset !== 0 || v.byteLength !== buf.byteLength) continue;
    if (seen.has(buf)) continue;
    seen.add(buf);
    out.push(buf);
  }
  return out;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let genie3Mod: any = null;
let gpuDevice: GPUDevice | null = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let weightBundle: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let generator: any = null;

async function ensureLoaded(id: number) {
  if (!genie3Mod) {
    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.02,
      detail: 'Loading Genie 3 WebGPU modules...',
    });
    const url = `${self.location.origin}/vendor/genie3/webgpu/index.js`;
    genie3Mod = await import(/* @vite-ignore */ url);
  }

  if (!gpuDevice) {
    const res = await genie3Mod.requestWebGpuDevice();
    if (!res?.device) {
      throw new Error('No compatible WebGPU device found for Genie 3');
    }
    const dev: GPUDevice = res.device;
    gpuDevice = dev;
    void dev.lost.then(() => {
      if (gpuDevice === dev) {
        gpuDevice = null;
        generator = null;
      }
    });
  }

  if (!generator) {
    if (!weightBundle) {
      self.postMessage({
        id,
        type: 'progress',
        phase: 'weights',
        progress: 0.05,
        detail: 'Fetching Genie 3 weights (~45 MB, cached after first run)...',
      });
      const envWeightsUrl = (import.meta.env.VITE_GENIE3_WEIGHTS_URL as string | undefined)?.trim();
      const weightsUrl =
        envWeightsUrl || 'https://huggingface.co/ramithuh/genie3-webgpu/resolve/main/fp16';
      weightBundle = await genie3Mod.fetchGenie3Weights(weightsUrl, {
        cache: true,
        onProgress: (p: { loaded?: number; total?: number; shardIndex?: number; totalShards?: number; shard?: string }) => {
          const loaded = p.loaded ?? 0;
          const total = p.total ?? 45 * 1024 * 1024;
          const frac = total > 0 ? loaded / total : 0.2;
          const mb = (loaded / 1024 / 1024).toFixed(1);
          const totMb = (total / 1024 / 1024).toFixed(1);
          self.postMessage({
            id,
            type: 'progress',
            phase: 'weights',
            progress: Math.min(0.92, frac),
            detail: `Genie 3 weights: ${mb} / ${totMb} MB`,
          });
        },
      });
    }

    self.postMessage({
      id,
      type: 'progress',
      phase: 'init',
      progress: 0.94,
      detail: 'Uploading Genie 3 weights to GPU...',
    });
    generator = await genie3Mod.Genie3GeneratorGpu.create(
      gpuDevice,
      weightBundle,
      weightBundle.config ?? null,
      { half: true },
    );
    // Drop CPU-side weight buffers immediately after GPU upload to reclaim ~45–90 MB JS heap
    weightBundle = null;
  }
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data;
  try {
    if (msg.type === 'dispose') {
      try {
        generator?.destroy?.();
      } catch {
        /* ignore */
      }
      generator = null;
      weightBundle = null;
      gpuDevice?.destroy();
      gpuDevice = null;
      self.postMessage({ id: msg.id, type: 'disposed' });
      return;
    }

    if (msg.type === 'run') {
      const t0 = performance.now();
      const p = msg.params ?? {};
      const length = Math.max(30, Math.min(256, Math.round(Number(p.length ?? 80))));
      const nSampleStep = Math.max(10, Math.min(250, Math.round(Number(p.nSampleStep ?? 100))));
      const directionScale = Math.max(0.0, Math.min(1.5, Number(p.directionScale ?? 0.8)));
      const noiseScale = Math.max(0, Math.min(1.5, Number(p.noiseScale ?? 1.0)));
      const predictSequence = p.predictSequence !== false;
      const seed = Math.max(0, Math.round(Number(p.seed ?? 42)));
      const condArtifact = msg.inputs?.condition as ConditionSpecArtifact | undefined;
      const template = (msg.inputs?.structure ?? msg.inputs?.template) as StructureArtifact | undefined;
      const taskMode = String(p.taskMode ?? 'unconditional');
      const inlineContig = String(p.contig ?? '').trim();
      const inlineLock = String(p.lockChain ?? '').trim().toUpperCase();
      const inlineHotspots = String(p.hotspots ?? '').trim();
      const inlineMotif = String(p.motifSpan ?? '').trim();

      // Fail fast BEFORE downloading 45 MB weights if required motif/binder inputs are missing
      if (!condArtifact && taskMode === 'motif' && !inlineMotif && !inlineContig) {
        throw new Error(
          'Genie 3 motif mode requires a motifSpan (e.g. A16-35) or contig specifying motif residues.',
        );
      }
      if (
        (taskMode === 'motif' || taskMode === 'binder' || Boolean(inlineLock) || Boolean(inlineMotif)) &&
        !template?.cifText &&
        !template?.pdbText &&
        !condArtifact?.inputCif
      ) {
        throw new Error(
          `Genie 3 "${taskMode}" mode requires a wired structure input (e.g. from PDB Input).`,
        );
      }

      // Guard against non-protein/enzyme/DNA templates across ALL Genie 3 modes (including inline motif/binder)
      if (template && (taskMode !== 'unconditional' || !condArtifact)) {
        if (template.benchmarkId) {
          const bench = getBenchmarkById(template.benchmarkId);
          if (bench && !bench.supportsGenie3) {
            throw new Error(
              `${bench.shortLabel} requires all-atom enzyme/ligand/DNA conditioning, which is supported by RFD4-Proteina rather than Genie 3.`,
            );
          }
        }
        if (template.diagnostics?.hasLigands || template.diagnostics?.hasNucleicAcids) {
          const extras = [
            ...(template.diagnostics.hasLigands ? [`ligands (${template.diagnostics.ligandNames.join(', ')})`] : []),
            ...(template.diagnostics.hasNucleicAcids ? [`nucleic acid chains (${template.diagnostics.nucleicAcidChains.join(', ')})`] : []),
          ];
          throw new Error(
            `Genie 3 is a protein-only Cα backbone diffusion model and cannot condition on ${extras.join(' and ')}. Use RFD4-Proteina for all-atom ligand/DNA conditioning.`,
          );
        }
      }

      await ensureLoaded(msg.id);

      let batch: Record<string, unknown> | undefined;
      let compiledSpec: Genie3CompiledSpec | undefined;
      let compiledSummary = '';
      let expectedChainLengths: number[] | undefined;

      let effectiveCond: ConditionSpecArtifact | undefined = condArtifact;
      if (!effectiveCond && (taskMode === 'binder' || taskMode === 'motif' || inlineLock || inlineMotif)) {
        const isBinder = taskMode === 'binder' || Boolean(inlineLock);
        const defaultChain = (template?.chainIds?.[0] || 'A').toUpperCase();
        const resolvedLock = isBinder ? inlineLock || defaultChain : '';
        const resolvedContig =
          inlineContig ||
          (isBinder ? `/0,${length}-${length}` : `${length}-${length}`);
        const intent = buildCanonicalConditionIntent({
          contig: resolvedContig,
          lockChain: resolvedLock,
          hotspots: inlineHotspots,
          motifSpan: inlineMotif,
        });
        effectiveCond = {
          kind: 'ConditionSpec',
          id: `inline-g3-${fnv1a(JSON.stringify({ intent, templateId: template?.id }))}`,
          label: isBinder ? `Binder (${resolvedLock})` : 'Motif Scaffold',
          contig: intent.contig,
          conditions: {},
          intent,
          inputCif: template?.cifText ?? template?.pdbText,
          specMode: true,
          summary: isBinder ? `binder vs ${resolvedLock}` : `motif ${inlineMotif}`,
        };
      } else if (!effectiveCond && template) {
        effectiveCond = buildDefaultGenie3ConditionFromTemplate(template, length);
      }

      if (effectiveCond) {
        compiledSpec = compileConditionToGenie3Spec(effectiveCond, template, length);
        batch = genie3Mod.createFeaturesFromSpec({
          length: compiledSpec.length,
          asymId: compiledSpec.asymId,
          symId: compiledSpec.symId,
          entityId: compiledSpec.entityId,
          residueIndex: compiledSpec.residueIndex,
          condGroup: compiledSpec.condGroup,
          condSeqMask: compiledSpec.condSeqMask,
          condStructMask: compiledSpec.condStructMask,
          condInterfaceMask: compiledSpec.condInterfaceMask,
          gtRestype: compiledSpec.gtRestype,
          gtAtomPositions: compiledSpec.gtAtomPositions,
        });
        compiledSummary = compiledSpec.summary;
        expectedChainLengths = compiledSpec.chainLengths;
      }

      const effectiveSteps = effectiveGenie3DdimSteps(nSampleStep);
      const stepNote =
        effectiveSteps !== nSampleStep ? ` (requested ${nSampleStep}, effective ${effectiveSteps})` : '';

      const res = await generator.generate({
        ...(batch ? { batch } : { length }),
        nSampleStep,
        directionScale,
        noiseScale,
        eta: 1.0,
        predictSequence,
        sequenceTemperature: 0.1,
        seed,
        // Emit honest Cα-only PDB/mmCIF (Genie 3 diffuses Cα coordinates only; never fabricate N/C/O)
        caOnly: true,
        yieldBetweenSteps: true,
        onStep: (info: { stepIndex: number; totalSteps: number; timestep: number }) => {
          self.postMessage({
            id: msg.id,
            type: 'progress',
            phase: 'sampling',
            progress: (info.stepIndex + 1) / Math.max(1, info.totalSteps),
            detail: `Genie 3 DDIM step ${info.stepIndex + 1}/${info.totalSteps}${stepNote} (t=${info.timestep})`,
          });
        },
      });

      let pdbText: string = res.pdb;
      let cifText: string = res.cif;
      let finalSeq: string;

      if (!predictSequence) {
        // Upstream generate.js falls back to unmasked ddimRes.seqLogits when predictSequence=false,
        // corrupting locked target/motif residues. Emit 'A' for unconditioned residues and preserve
        // gtRestype wherever condSeqMask > 0.5, then re-serialize honest Cα-only PDB/mmCIF cleanly.
        const outLen = Number(res.length ?? compiledSpec?.length ?? length);
        finalSeq = resolveGenie3PolyAlaSequence(outLen, compiledSpec);
        if (res.backbone4 && typeof genie3Mod.backboneToPdb === 'function') {
          const serializeOpts = {
            sequence: finalSeq,
            chainId: 'A',
            chainIds: (batch as { asym_id?: ArrayLike<number> } | undefined)?.asym_id ?? null,
            residueIndices: (batch as { residue_index?: ArrayLike<number> } | undefined)?.residue_index ?? null,
            caOnly: true,
          };
          pdbText = genie3Mod.backboneToPdb(res.backbone4, outLen, serializeOpts);
          if (typeof genie3Mod.backboneToMmcif === 'function') {
            cifText = genie3Mod.backboneToMmcif(res.backbone4, outLen, {
              entryId: 'GENIE3',
              ...serializeOpts,
            });
          }
        }
      } else {
        const parsedRaw = parsePdbBackbone(pdbText);
        finalSeq =
          typeof res.sequence === 'string' && res.sequence.length === parsedRaw.nResidues
            ? res.sequence
            : parsedRaw.sequence;
      }

      const parsed = parsePdbBackbone(pdbText);
      const geometry = computeGeometryStats(
        parsed.caCoords,
        parsed.chainIds,
        parsed.resNums,
        parsed.insCodes,
      );
      const candChainLengths =
        expectedChainLengths && expectedChainLengths.length > 1 &&
        expectedChainLengths.reduce((a, b) => a + b, 0) === finalSeq.length
          ? expectedChainLengths
          : parsed.diagnostics.chainLengths.length > 1
            ? parsed.diagnostics.chainLengths
            : undefined;

      // Emit lockedChains, fixedResidueIndices, chainRoles, motifIndices, and nativeMotifCoords
      let lockedChains: string[] | undefined;
      let fixedResidueIndices: number[] | undefined;
      let chainRoles: Record<string, 'target' | 'binder' | 'scaffold'> | undefined;
      let motifIndices: number[] | undefined;
      let nativeMotifCoords: number[][] | undefined;

      if (compiledSpec?.mode === 'binder') {
        // Genie 3 binder-first convention: chain A (asym 0) = binder, chain B (asym 1) = fixed target
        lockedChains = ['B'];
        chainRoles = { A: 'binder', B: 'target' };
      } else if (compiledSpec?.mode === 'motif') {
        chainRoles = { A: 'scaffold' };
        const fixed1Based: number[] = [];
        const motif0Based: number[] = [];
        const nativeCoords: number[][] = [];
        for (let i = 0; i < compiledSpec.length; i++) {
          if (compiledSpec.condStructMask[i] > 0.5 || compiledSpec.condSeqMask[i] > 0.5) {
            fixed1Based.push(i + 1);
            if (compiledSpec.condStructMask[i] > 0.5) {
              motif0Based.push(i);
              nativeCoords.push([
                compiledSpec.gtAtomPositions[i * 3 + 0],
                compiledSpec.gtAtomPositions[i * 3 + 1],
                compiledSpec.gtAtomPositions[i * 3 + 2],
              ]);
            }
          }
        }
        if (fixed1Based.length > 0) fixedResidueIndices = fixed1Based;
        if (motif0Based.length > 0) {
          motifIndices = motif0Based;
          nativeMotifCoords = nativeCoords;
        }
      } else {
        chainRoles = { A: 'scaffold' };
      }

      const runHash = fnv1a(
        JSON.stringify({
          length,
          nSampleStep,
          directionScale,
          noiseScale,
          predictSequence,
          seed,
          condId: effectiveCond?.id ?? null,
          templateId: template?.id ?? null,
          taskMode,
          inlineContig,
          inlineLock,
          inlineHotspots,
          inlineMotif,
        }),
      );

      const structArtifact: StructureArtifact = {
        kind: 'StructureArtifact',
        id: `genie3-L${parsed.nResidues}-s${seed}-${runHash}`,
        label: compiledSummary
          ? `Genie 3 (${compiledSummary}, ${res.stepsRun} steps)`
          : `Genie 3 (L=${parsed.nResidues}, ${res.stepsRun} DDIM steps)`,
        sourceModelId: 'genie3',
        length: parsed.nResidues,
        sequence: finalSeq,
        caCoords: parsed.caCoords,
        backbone4Coords: parsed.backbone4Coords,
        pdbText,
        cifText,
        chainIds: parsed.chainIds,
        chainLengths: candChainLengths ?? parsed.diagnostics.chainLengths,
        insCodes: parsed.insCodes,
        sourceResidues: parsed.sourceResidues,
        chainIdMap: parsed.chainIdMap,
        atomPresence: parsed.atomPresence,
        observedBackboneMask: parsed.observedBackboneMask,
        diagnostics: parsed.diagnostics,
        ...(lockedChains ? { lockedChains } : {}),
        ...(fixedResidueIndices ? { fixedResidueIndices } : {}),
        ...(chainRoles ? { chainRoles } : {}),
        ...(motifIndices ? { motifIndices, nativeMotifCoords } : {}),
        geometry,
      };

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `genie3-seq-${structArtifact.id}`,
        parentStructureId: structArtifact.id,
        sourceModelId: predictSequence ? 'genie3 (1-step SequenceNet)' : 'genie3 (poly-ALA backbone)',
        candidates: [
          {
            id: 'genie3-native-1',
            sequence: finalSeq,
            score: null,
            globalScore: null,
            sourceModelId: 'genie3',
            seed,
            ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
          },
        ],
      };

      const transferables = collectOwnedTransferables([
        structArtifact.caCoords,
        structArtifact.backbone4Coords,
        structArtifact.observedBackboneMask,
      ]);
      self.postMessage(
        {
          id: msg.id,
          type: 'result',
          outputs: { structure: structArtifact, sequences: seqSet },
          elapsedMs: performance.now() - t0,
        },
        transferables,
      );
    }
  } catch (err) {
    self.postMessage({
      id: msg.id,
      type: 'error',
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
