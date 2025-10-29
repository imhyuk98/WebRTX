import { Camera } from './camera';
import { Controls } from './controls';
import type { FallbackSceneDescriptor, SphereInstance } from './fallback/scene_descriptor';

function createDefaultSceneDescriptor(): FallbackSceneDescriptor {
  return {
    spheres: [{ center: [0, 0, 0], radius: 1.2 }],
    cylinders: [{
      center: [1.8, 0.0, 0.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 0.0, -1.0],
      radius: 0.4,
      height: 2.0,
      angleDeg: 360.0,
    }],
    circles: [{
      center: [-1.8, 0.0, 0.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 1.0, 0.0],
      radius: 0.9,
    }],
    ellipses: [{
      center: [-4.0, 0.0, 0.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 1.0, 0.0],
      radiusX: 0.9,
      radiusY: 0.5,
    }],
    cones: [{
      center: [0.0, -1.0, -1.2],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 0.0, 1.0],
      radius: 0.7,
      height: 1.6,
    }],
    lines: [{
      p0: [-2.8, 0.8, -0.5],
      p1: [-1.6, 1.6, -0.2],
      radius: 0.001,
    }],
    tori: [{
      center: [4.0, 0.2, 0.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 1.0, 0.0],
      majorRadius: 0.9,
      minorRadius: 0.25,
      angleDeg: 360.0,
    }],
    planes: [{
      center: [0.0, -2.0, 0.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 0.0, 1.0],
      halfWidth: 10.0,
      halfHeight: 10.0,
    }],
    bezierPatches: [],
  };
}

function createSphereGridBaseDescriptor(): FallbackSceneDescriptor {
  return {
    spheres: [],
    cylinders: [],
    circles: [],
    ellipses: [],
    cones: [],
    lines: [],
    tori: [],
    planes: [{
      center: [0.0, -2.0, 0.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 0.0, 1.0],
      halfWidth: 50.0,
      halfHeight: 50.0,
    }],
    bezierPatches: [],
  };
}

function generateSphereGridInstances(nx: number, ny: number, nz: number, radius = 0.1, spacing = radius * 2.5): SphereInstance[] {
  const total = nx * ny * nz;
  const spheres: SphereInstance[] = new Array(total);
  const offsetX = (nx - 1) * 0.5 * spacing;
  const offsetY = (ny - 1) * 0.5 * spacing;
  const offsetZ = (nz - 1) * 0.5 * spacing;
  let i = 0;
  for (let x = 0; x < nx; x++) {
    const cx = x * spacing - offsetX;
    for (let y = 0; y < ny; y++) {
      const cy = y * spacing - offsetY;
      for (let z = 0; z < nz; z++) {
        const cz = z * spacing - offsetZ;
        spheres[i++] = {
          center: [cx, cy, cz],
          radius,
        };
      }
    }
  }
  return spheres;
}

export interface InteractiveOptions {
  width?: number; height?: number;
  fovYDeg?: number;
  moveSpeed?: number;
  mouseSensitivity?: number;
  initialYawDeg?: number;
  initialPitchDeg?: number;
  invertY?: boolean;
}

export async function runInteractiveScene(canvas: HTMLCanvasElement | string, opts: InteractiveOptions = {}) {
  const { width = 1920, height = 1080 } = opts;
  const { runMinimalScene } = await import('./scene');
  const result = await runMinimalScene(canvas as any, width, height);
  const canvasEl: HTMLCanvasElement = (typeof canvas === 'string') ? document.getElementById(canvas) as HTMLCanvasElement : canvas;
  const cam = new Camera([0,0,3],[0,0,0],[0,1,0]);
  const controls = new Controls(canvasEl, cam, { yawDeg: opts.initialYawDeg, pitchDeg: opts.initialPitchDeg, invertY: opts.invertY });
  if (opts.moveSpeed) controls.setMoveSpeed(opts.moveSpeed);
  if (opts.mouseSensitivity) controls.setMouseSensitivity(opts.mouseSensitivity);
  let activeSceneIndex = 0;
  let switchingScene = false;
  let cachedSphereGrid: SphereInstance[] | null = null;
  const applyScene = async (sceneIndex: number) => {
    if (switchingScene || sceneIndex === activeSceneIndex) {
      return;
    }
    switchingScene = true;
    try {
      if (sceneIndex === 0) {
        await result.setScene(createDefaultSceneDescriptor());
      } else if (sceneIndex === 1) {
        const baseDescriptor = createSphereGridBaseDescriptor();
        await result.setScene(baseDescriptor);
        if (!cachedSphereGrid) {
          console.time('[Interactive] Generating 20x20x20 sphere grid');
          cachedSphereGrid = generateSphereGridInstances(20, 20, 20, 0.12, 0.35);
          console.timeEnd('[Interactive] Generating 20x20x20 sphere grid');
        }
        await result.setSpheres(cachedSphereGrid);
      } else {
        return;
      }
      activeSceneIndex = sceneIndex;
      await result.render();
      console.info('[Interactive] Scene switched to', sceneIndex + 1);
    } catch (err) {
      console.error('[Interactive] Failed to switch scene', sceneIndex + 1, err);
    } finally {
      switchingScene = false;
    }
  };
  const handleSceneKey = (event: KeyboardEvent) => {
    if (event.code === 'Digit1') {
      event.preventDefault();
      applyScene(0);
    } else if (event.code === 'Digit2') {
      event.preventDefault();
      applyScene(1);
    }
  };
  window.addEventListener('keydown', handleSceneKey);
  console.log('[Interactive] Scene hotkeys: [1] default demo, [2] 20x20x20 sphere grid');
  let last = performance.now();
  let stopped = false;
  async function frame() {
    if (stopped) return;
    const now = performance.now();
    const dt = (now - last)/1000; last = now;
    controls.update(dt);
    await result.updateCamera(cam.position, cam.look_at, cam.up);
    await result.render();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  function dispose() {
    stopped = true;
    window.removeEventListener('keydown', handleSceneKey);
  }
  return { ...result, camera: cam, controls, dispose };
}
