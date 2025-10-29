import { Camera } from './camera';
import { Controls } from './controls';
import type { FallbackSceneDescriptor, SphereInstance } from './fallback/scene_descriptor';

function createDefaultSceneDescriptor(): FallbackSceneDescriptor {
  return {
    spheres: [{ center: [-9.0, -0.2, -3.0], radius: 1.1 }],
    cylinders: [{
      center: [-6.0, -1.0, -3.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 0.0, -1.0],
      radius: 0.65,
      height: 2.8,
      angleDeg: 360.0,
    }],
    circles: [{
      center: [-3.0, 0.0, -3.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 1.0, 0.0],
      radius: 1.3,
    }],
    ellipses: [{
      center: [0.0, 0.0, -3.0],
      xdir: [0.96, 0.20, 0.08],
      ydir: [-0.10, 0.96, 0.25],
      radiusX: 1.4,
      radiusY: 0.7,
    }],
    cones: [{
      center: [3.0, -1.4, -3.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 0.0, 1.0],
      radius: 0.9,
      height: 2.4,
    }],
    lines: [{
      p0: [5.8, -0.8, -3.0],
      p1: [6.6, 1.4, -3.0],
      radius: 0.025,
    }],
    tori: [{
      center: [9.0, -0.1, -3.0],
      xdir: [0.97, 0.04, 0.23],
      ydir: [-0.06, 0.98, 0.18],
      majorRadius: 1.3,
      minorRadius: 0.32,
      angleDeg: 360.0,
    }],
    planes: [{
      center: [0.0, -3.8, -3.0],
      xdir: [1.0, 0.0, 0.0],
      ydir: [0.0, 0.0, 1.0],
      halfWidth: 45.0,
      halfHeight: 45.0,
    }],
    bezierPatches: [{
      p00: [-2.0, -1.2, 2.5],
      p10: [-0.5, -0.8, 3.2],
      p01: [0.5, -0.6, 2.0],
      p11: [1.8, -0.5, 2.8],
      du00: [2.2, 0.3, 0.5],
      du10: [2.4, 0.5, 0.9],
      du01: [2.2, 0.6, -0.3],
      du11: [2.1, 0.7, -0.2],
      dv00: [0.3, 0.2, -1.9],
      dv10: [-0.5, 0.5, -2.1],
      dv01: [0.6, 0.3, -1.7],
      dv11: [-0.6, 0.4, -2.0],
      duv00: [0.2, 0.1, 0.3],
      duv10: [-0.2, 0.1, -0.3],
      duv01: [0.3, 0.0, 0.4],
      duv11: [-0.3, 0.1, 0.3],
      maxDepth: 6,
      pixelEpsilon: 2.0,
    }],
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
        baseDescriptor.planes = [];
          await result.setScene(baseDescriptor);
        if (!cachedSphereGrid) {
          console.time('[Interactive] Generating 100x100x100 sphere grid');
          cachedSphereGrid = generateSphereGridInstances(100, 100, 100, 0.12, 0.35);
          console.timeEnd('[Interactive] Generating 100x100x100 sphere grid');
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
  console.log('[Interactive] Scene hotkeys: [1] default demo, [2] 100x100x100 sphere grid');
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
