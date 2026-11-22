/// <reference lib="webworker" />
// Dedicated WebGPU worker wrapping ramithuh/genie3-webgpu (Apache-2.0 port of aqlaboratory/genie3)
// Generates de novo protein backbones (ideal N/CA/C/O + Cα) and optional 1-step native sequences
// via SE(3)-equivariant DDIM diffusion on WebGPU (~45 MB FP16 weights).

import { parsePdbBackbone } from '$lib/protein-ops/pdb';
import { computeGeometryStats } from '$lib/protein-ops/kabsch';
import {
  resolveGenie3CompiledInput,
  resolveGenie3PolyAlaSequence,
} from '$lib/engine/conditioning';
import { makeSequenceKey } from '$lib/engine/collections';
import { deriveStableItemSeed, planGeneratorDesigns } from '$lib/engine/work-plan';
import type {
  ConditionSpecArtifact,
  SequenceCandidate,
  SequenceSet,
  StructureArtifact,
} from '$lib/engine/types';

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
        onProgress: (p: {
          loaded?: number;
          total?: number;
          loadedBytes?: number;
          totalBytes?: number;
          shardIndex?: number;
          totalShards?: number;
          shard?: string;
        }) => {
          const loaded = p.loadedBytes ?? p.loaded ?? 0;
          const total = p.totalBytes ?? p.total ?? 45 * 1024 * 1024;
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
      const condArtifact = msg.inputs?.condition as ConditionSpecArtifact | undefined;
      const template = (msg.inputs?.structure ?? msg.inputs?.template) as StructureArtifact | undefined;
      // Keep explicit p.<key> reads intact for param-schema bidirectional coverage guards
      const rawNumDesigns = Number(p.numDesigns ?? 1);
      const numDesigns = Number.isFinite(rawNumDesigns)
        ? Math.max(1, Math.min(16, Math.round(rawNumDesigns)))
        : 1;
      const rawParams: Record<string, unknown> = {
        length: p.length,
        numDesigns,
        nSampleStep: p.nSampleStep,
        directionScale: p.directionScale,
        noiseScale: p.noiseScale,
        predictSequence: p.predictSequence,
        seed: p.seed,
        taskMode: p.taskMode,
        contig: p.contig,
        lockChain: p.lockChain,
        hotspots: p.hotspots,
        motifSpan: p.motifSpan,
      };
      const taskMode = String(rawParams.taskMode ?? 'unconditional');
      const inlineContig = String(rawParams.contig ?? '').trim();
      const inlineLock = String(rawParams.lockChain ?? '').trim().toUpperCase();
      const inlineHotspots = String(rawParams.hotspots ?? '').trim();
      const inlineMotif = String(rawParams.motifSpan ?? '').trim();

      // Fail fast BEFORE downloading 45 MB weights if required template structure is missing or non-protein
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
      if (
        template &&
        (taskMode !== 'unconditional' || !condArtifact) &&
        (template.diagnostics?.hasLigands || template.diagnostics?.hasNucleicAcids)
      ) {
        const extras = [
          ...(template.diagnostics.hasLigands ? [`ligands (${template.diagnostics.ligandNames.join(', ')})`] : []),
          ...(template.diagnostics.hasNucleicAcids ? [`nucleic acid chains (${template.diagnostics.nucleicAcidChains.join(', ')})`] : []),
        ];
        throw new Error(
          `Genie 3 is a protein-only Cα backbone diffusion model and cannot condition on ${extras.join(' and ')}. Use RFD4-Proteina for all-atom ligand/DNA conditioning.`,
        );
      }

      // Compile conditioning spec via single-source-of-truth resolver BEFORE downloading 45 MB weights
      const resolved = resolveGenie3CompiledInput(rawParams, template, condArtifact);
      const {
        length,
        nSampleStep,
        effectiveSteps,
        directionScale,
        noiseScale,
        predictSequence,
        seed,
        effectiveCond,
        compiledSpec,
      } = resolved;

      await ensureLoaded(msg.id);

      let batch: Record<string, unknown> | undefined;
      let compiledSummary = '';
      let expectedChainLengths: number[] | undefined;

      if (compiledSpec) {
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

      const stepNote =
        effectiveSteps !== nSampleStep ? ` (requested ${nSampleStep}, effective ${effectiveSteps})` : '';

      const runOneDesign = async (
        designIndex: number,
        totalDesigns: number,
        itemSeed: number,
        designKey?: string,
      ): Promise<{ struct: StructureArtifact; candidate: SequenceCandidate }> => {
        const res = await generator.generate({
          ...(batch ? { batch } : { length }),
          nSampleStep,
          directionScale,
          noiseScale,
          eta: 1.0,
          predictSequence,
          sequenceTemperature: 0.1,
          seed: itemSeed,
          // Emit honest Cα-only PDB/mmCIF (Genie 3 diffuses Cα coordinates only; never fabricate N/C/O)
          caOnly: true,
          yieldBetweenSteps: true,
          onStep: (info: { stepIndex: number; totalSteps: number; timestep: number }) => {
            const stepFrac = (info.stepIndex + 1) / Math.max(1, info.totalSteps);
            const progress =
              totalDesigns > 1 ? (designIndex + stepFrac) / totalDesigns : stepFrac;
            const detail =
              totalDesigns > 1
                ? `Design ${designIndex + 1}/${totalDesigns} · DDIM step ${info.stepIndex + 1}/${info.totalSteps}${stepNote} (t=${info.timestep})`
                : `Genie 3 DDIM step ${info.stepIndex + 1}/${info.totalSteps}${stepNote} (t=${info.timestep})`;
            self.postMessage({
              id: msg.id,
              type: 'progress',
              phase: 'sampling',
              progress,
              detail,
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
            seed: itemSeed,
            condId: effectiveCond?.id ?? null,
            templateId: template?.id ?? null,
            taskMode,
            inlineContig,
            inlineLock,
            inlineHotspots,
            inlineMotif,
          }),
        );

        const structId =
          totalDesigns > 1
            ? `genie3-d${designIndex + 1}-L${parsed.nResidues}-s${itemSeed}-${runHash}`
            : `genie3-L${parsed.nResidues}-s${itemSeed}-${runHash}`;
        const label =
          totalDesigns > 1
            ? compiledSummary
              ? `Genie 3 #${designIndex + 1}/${totalDesigns} (${compiledSummary}, ${res.stepsRun} steps)`
              : `Genie 3 #${designIndex + 1}/${totalDesigns} (L=${parsed.nResidues}, ${res.stepsRun} DDIM steps)`
            : compiledSummary
              ? `Genie 3 (${compiledSummary}, ${res.stepsRun} steps)`
              : `Genie 3 (L=${parsed.nResidues}, ${res.stepsRun} DDIM steps)`;

        const struct: StructureArtifact = {
          kind: 'StructureArtifact',
          id: structId,
          label,
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
          bfactorSemantics: 'none',
          atomGranularity: 'ca-only',
          ...(designKey ? { designKey, designIndex, seed: itemSeed } : {}),
          ...(lockedChains ? { lockedChains } : {}),
          ...(fixedResidueIndices ? { fixedResidueIndices } : {}),
          ...(chainRoles ? { chainRoles } : {}),
          ...(motifIndices ? { motifIndices, nativeMotifCoords } : {}),
          geometry,
        };

        const candidate: SequenceCandidate =
          totalDesigns > 1
            ? {
                id: `genie3-d${designIndex + 1}-native`,
                ...(designKey ? { designKey, designIndex } : {}),
                sourceStructureId: struct.id,
                sampleIndex: designIndex,
                sequence: finalSeq,
                score: null,
                globalScore: null,
                sourceModelId: 'genie3',
                seed: itemSeed,
                ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
              }
            : {
                id: 'genie3-native-1',
                sequence: finalSeq,
                score: null,
                globalScore: null,
                sourceModelId: 'genie3',
                seed: itemSeed,
                ...(candChainLengths ? { chainLengths: candChainLengths } : {}),
              };

        return { struct, candidate };
      };

      if (numDesigns === 1) {
        // Legacy singleton path: exact seed, artifact ID, and singleton StructureArtifact semantics
        const { struct: structArtifact, candidate } = await runOneDesign(0, 1, seed);
        const seqSet: SequenceSet = {
          kind: 'SequenceSet',
          id: `genie3-seq-${structArtifact.id}`,
          parentStructureId: structArtifact.id,
          sourceStructureId: structArtifact.id,
          sourceModelId: predictSequence ? 'genie3 (1-step SequenceNet)' : 'genie3 (poly-ALA backbone)',
          ...(structArtifact.lockedChains ? { lockedChains: structArtifact.lockedChains } : {}),
          ...(structArtifact.chainRoles ? { chainRoles: structArtifact.chainRoles } : {}),
          candidates: [candidate],
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
        return;
      }

      // Multi-design resident-weight sequential loop (weights loaded ONCE above)
      const plan = planGeneratorDesigns({
        modelId: 'genie3',
        numDesigns,
        baseSeed: seed,
        scientificParams: {
          length,
          nSampleStep,
          directionScale,
          noiseScale,
          predictSequence,
          taskMode,
          contig: inlineContig,
          lockChain: inlineLock,
          hotspots: inlineHotspots,
          motifSpan: inlineMotif,
        },
        inputDigest: fnv1a(
          JSON.stringify({
            condId: effectiveCond?.id ?? null,
            templateId: template?.id ?? null,
          }),
        ),
      });

      const designs: StructureArtifact[] = [];
      const candidates: SequenceCandidate[] = [];
      for (let d = 0; d < plan.items.length; d++) {
        const planned = plan.items[d];
        const itemSeed = deriveStableItemSeed(seed, 'generation', planned.key);
        const { struct, candidate } = await runOneDesign(d, numDesigns, itemSeed, planned.key);
        const seqKey = makeSequenceKey(plan.familyKey, planned.key, 0);
        candidate.key = seqKey;
        candidate.sequenceKey = seqKey;
        designs.push(struct);
        candidates.push(candidate);
      }

      const representative = designs[0];
      const repSelection = {
        selectedKey: representative.designKey ?? representative.id,
        criterion: 'lowest-creation-ordinal',
        totalMembers: designs.length,
        selectedIndex: 0,
        scoreValue: null,
      };
      const structArtifact: StructureArtifact = {
        ...representative,
        label: compiledSummary
          ? `Genie 3 (${numDesigns} designs · ${compiledSummary})`
          : `Genie 3 (${numDesigns} designs · L=${representative.length})`,
        representativeSelection: repSelection,
        collection: {
          kind: 'Collection',
          schema: 'denovo.collection/v1',
          id: `genie3-col-${plan.planDigest.slice(0, 16)}`,
          familyKey: plan.familyKey,
          collectionKind: 'independent-candidates',
          requestedCount: numDesigns,
          complete: true,
          items: designs,
          keys: designs.map((item) => item.designKey ?? item.id),
          representativeSelection: repSelection,
        },
      };

      const seqSet: SequenceSet = {
        kind: 'SequenceSet',
        id: `genie3-seq-col-${plan.planDigest.slice(0, 16)}`,
        parentStructureId: representative.id,
        sourceStructureId: representative.id,
        sourceStructureIds: designs.map((item) => item.id),
        sourceModelId: predictSequence ? 'genie3 (1-step SequenceNet)' : 'genie3 (poly-ALA backbone)',
        collectionKind: 'independent-candidates',
        ...(representative.lockedChains ? { lockedChains: representative.lockedChains } : {}),
        ...(representative.chainRoles ? { chainRoles: representative.chainRoles } : {}),
        candidates,
      };

      const transferViews: Array<ArrayBufferView | undefined> = [];
      for (const item of designs) {
        transferViews.push(item.caCoords, item.backbone4Coords, item.observedBackboneMask);
      }
      const transferables = collectOwnedTransferables(transferViews);
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
