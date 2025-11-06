// Load the polyfill first so globals like GPUShaderStageRTX, GPUBufferUsageRTX exist before other imports
import './patch';
export * from './types';
export { initBvhWasm } from './wasm_bvh_builder';
export { runMinimalScene, runSphereScene, runPlaneScene, runRayTracingScene } from './scene';
export { Camera } from './camera';
export { Controls } from './controls';
import { attachDTDSSceneLoader, SceneSettable } from './importer/world_to_scene';

type SceneHandleWithRender = SceneSettable & { render?: () => Promise<void> };

const DTDS_INPUT_ID = 'dtdsPicker';
const attachedSceneHandles = new WeakSet<SceneHandleWithRender>();

function tryAttachDtdsLoader(sceneHandle: SceneHandleWithRender): void {
	if (typeof document === 'undefined' || !sceneHandle) {
		return;
	}
	if (attachedSceneHandles.has(sceneHandle)) {
		return;
	}
	const input = document.getElementById(DTDS_INPUT_ID) as HTMLInputElement | null;
	if (!input) {
		return;
	}
	attachDTDSSceneLoader(input, sceneHandle, {
		afterSetScene: async () => {
			if (typeof sceneHandle.render === 'function') {
				try {
					await sceneHandle.render();
				} catch (renderError) {
					console.warn('[WebRTX] Failed to render DTDS scene after import.', renderError);
				}
			}
		},
	});
	attachedSceneHandles.add(sceneHandle);
}

// Optional global helper for manual console start if user loaded only dist bundle
if (typeof window !== 'undefined') {
	let interactiveStarting = false;
	(window as any).startInteractive = async (canvasId: string = 'gfx', opts: any = {}) => {
		if ((window as any).interactive || interactiveStarting) {
			console.log('[WebRTX] Interactive already starting/started, skipping');
			return (window as any).interactive;
		}
		interactiveStarting = true;
		try {
			const { runInteractiveScene } = await import('./interactive');
			const res = await runInteractiveScene(canvasId, opts);
			(window as any).interactive = res;
			tryAttachDtdsLoader(res as SceneHandleWithRender);
			console.log('[WebRTX] Interactive scene started. Access via window.interactive');
			return res;
		} finally {
			interactiveStarting = false;
		}
	};
	console.log('[WebRTX] window.startInteractive(canvasId, opts) available');

	// Quick console helpers to tweak primitives at runtime (require interactive scene)
	(window as any).setCylinder = async (center:[number,number,number], xdir:[number,number,number], ydir:[number,number,number], radius:number, height:number, angleDeg?:number) => {
		const h = (window as any).interactive; if (!h?.setCylinder) { console.warn('[WebRTX] interactive.setCylinder not available'); return; }
		await h.setCylinder({ center, xdir, ydir, radius, height, angleDeg }); await h.render();
	};
	(window as any).setTorus = async (center:[number,number,number], xdir:[number,number,number], ydir:[number,number,number], majorR:number, minorR:number, angleDeg?: number) => {
		const h = (window as any).interactive; if (!h?.setTorus) { console.warn('[WebRTX] interactive.setTorus not available'); return; }
		await h.setTorus({ center, xdir, ydir, majorR, minorR, angleDeg }); await h.render();
	};

	// Multi-instance torus helper
	(window as any).setTori = async (list: Array<{ center:[number,number,number]; xdir:[number,number,number]; ydir:[number,number,number]; majorR:number; minorR:number; angleDeg?:number; }>) => {
		const h = (window as any).interactive; if (!h?.setTori) { console.warn('[WebRTX] interactive.setTori not available'); return; }
		await h.setTori(list); await h.render();
	};

	// Optional helper: set line using startPoint/endPoint and optional thickness
	(window as any).setLine = (startPoint: [number,number,number], endPoint: [number,number,number], thickness?: number) => {
		(globalThis as any).__webrtxOverrides = {
			startPoint, endPoint,
			...(typeof thickness === 'number' ? { thickness } : {})
		};
		console.log('[WebRTX] Line override set', { startPoint, endPoint, thickness });
	};

	window.addEventListener('DOMContentLoaded', async () => {
		const canvas: any = document.getElementById('gfx') || document.querySelector('canvas[data-webrtx]');
		if (canvas && (canvas.hasAttribute('data-autointeractive') || canvas.getAttribute('data-mode') === 'interactive')) {
			if (!(window as any).interactive) {
				console.log('[WebRTX] Auto launching interactive mode');
					try {
						// Defer a microtask to ensure all modules/patches finished evaluating
						await Promise.resolve();
						const handle = await (window as any).startInteractive(canvas.id || 'gfx');
						tryAttachDtdsLoader(handle as SceneHandleWithRender);
					} catch (e) { console.warn('[WebRTX] Auto interactive failed', e); }
			}
		}
	});
}