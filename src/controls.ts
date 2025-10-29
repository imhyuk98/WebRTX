import { Camera } from './camera';

export class Controls {
  canvas: HTMLCanvasElement;
  camera: Camera;
  keysPressed: Set<string> = new Set();
  mouseLocked = false;
  yaw = 0; // degrees
  pitch = 0; // degrees
  moveSpeed = 5.0;
  mouseSensitivity = 0.15;
  invertY = false;
  private lastTime = performance.now();

  constructor(canvas: HTMLCanvasElement, camera: Camera, opts?: { yawDeg?: number; pitchDeg?: number; invertY?: boolean; }) {
    this.canvas = canvas;
    this.camera = camera;
    if (opts?.yawDeg !== undefined) this.yaw = opts.yawDeg;
    if (opts?.pitchDeg !== undefined) this.pitch = opts.pitchDeg;
    if (opts?.invertY !== undefined) this.invertY = opts.invertY;
    this.setupEventListeners();
    this.setupPointerLock();
    // initialize look_at based on yaw/pitch
    this.updateCameraRotation();
  }

  private setupEventListeners() {
    document.addEventListener('keydown', e => {
      this.keysPressed.add(e.code);
    });
    document.addEventListener('keyup', e => {
      this.keysPressed.delete(e.code);
    });
    document.addEventListener('mousemove', e => {
      if (this.mouseLocked) {
        this.handleMouseMove(e.movementX, e.movementY);
      }
    });
  }

  private setupPointerLock() {
    // Ensure canvas focusable
    if (!this.canvas.hasAttribute('tabindex')) this.canvas.setAttribute('tabindex','0');
    // Prevent default context menu which can interfere with lock attempts
    this.canvas.addEventListener('contextmenu', e => e.preventDefault());
    const requestLock = (reason?: string) => {
      // Some browsers require element to be focused
      (this.canvas as any).focus?.();
      try {
        const p: any = this.canvas.requestPointerLock();
        console.log('[Controls] requestPointerLock called', reason||'click');
        if (p && typeof p.catch === 'function') {
          p.catch((err: any)=> console.warn('[Controls] PointerLock request rejected:', err));
        }
      } catch (err) {
        console.warn('[Controls] requestPointerLock threw', err);
      }
    };
    this.canvas.addEventListener('click', () => requestLock('click'));
    this.canvas.addEventListener('pointerdown', (e) => {
      if (!this.mouseLocked && e.button === 0) requestLock('pointerdown');
    });
    document.addEventListener('pointerlockchange', () => {
      const locked = document.pointerLockElement === this.canvas;
      this.mouseLocked = locked;
      console.log('[Controls] pointerlockchange locked=', locked);
    });
    document.addEventListener('pointerlockerror', (e) => {
      console.warn('[Controls] pointerlockerror event', e);
    });
    document.addEventListener('keydown', e => {
      if (e.code === 'Escape' && this.mouseLocked) {
        document.exitPointerLock();
      }
      // Allow manual lock retry with L key
      if (e.code === 'KeyL' && !this.mouseLocked) requestLock('KeyL');
    });
  }

  private handleMouseMove(dx: number, dy: number) {
    this.yaw += dx * this.mouseSensitivity;
    // Browser event: moving mouse up gives negative dy. We want up => increase pitch.
    const baseDelta = -dy * this.mouseSensitivity; // up (dy<0) -> positive
    this.pitch += this.invertY ? -baseDelta : baseDelta;
    this.pitch = Math.max(-89, Math.min(89, this.pitch));
    this.updateCameraRotation();
  }

  private updateCameraRotation() {
    const yawRad = this.yaw * Math.PI / 180;
    const pitchRad = this.pitch * Math.PI / 180;
    const forward: [number,number,number] = [
      Math.cos(pitchRad) * Math.cos(yawRad),
      Math.sin(pitchRad),
      Math.cos(pitchRad) * Math.sin(yawRad)
    ];
    // Dynamic up reconstruction to avoid gimbal issues near poles.
    const worldUp: [number,number,number] = [0,1,0];
    // right = normalize(cross(forward, worldUp))
    let right: [number,number,number] = [
      forward[1]*worldUp[2]-forward[2]*worldUp[1],
      forward[2]*worldUp[0]-forward[0]*worldUp[2],
      forward[0]*worldUp[1]-forward[1]*worldUp[0]
    ];
    const rl = Math.hypot(right[0],right[1],right[2]);
    if (rl < 1e-4) {
      // forward almost parallel to worldUp; pick alternate world up (0,0,1)
      const alt: [number,number,number] = [0,0,1];
      right = [
        forward[1]*alt[2]-forward[2]*alt[1],
        forward[2]*alt[0]-forward[0]*alt[2],
        forward[0]*alt[1]-forward[1]*alt[0]
      ];
    }
    const invRl = 1/(Math.hypot(right[0],right[1],right[2])||1);
    right = [right[0]*invRl,right[1]*invRl,right[2]*invRl];
    // up = normalize(cross(right, forward))
    let up: [number,number,number] = [
      right[1]*forward[2]-right[2]*forward[1],
      right[2]*forward[0]-right[0]*forward[2],
      right[0]*forward[1]-right[1]*forward[0]
    ];
    const ul = 1/(Math.hypot(up[0],up[1],up[2])||1);
    up = [up[0]*ul, up[1]*ul, up[2]*ul];
    this.camera.up = up;
    this.camera.setLookAt(
      this.camera.position[0] + forward[0],
      this.camera.position[1] + forward[1],
      this.camera.position[2] + forward[2]
    );
  }

  update(deltaTime: number) {
    const dist = this.moveSpeed * deltaTime;
    const fwd = this.getForward();
    const right = this.getRight();

    if (this.keysPressed.has('KeyW')) this.move(fwd, dist);
    if (this.keysPressed.has('KeyS')) this.move(fwd, -dist);
    if (this.keysPressed.has('KeyD')) this.move(right, dist);
    if (this.keysPressed.has('KeyA')) this.move(right, -dist);

    if (this.keysPressed.has('Space')) {
      const up = this.normalize(this.camera.up);
      this.camera.position[0]+=up[0]*dist; this.camera.position[1]+=up[1]*dist; this.camera.position[2]+=up[2]*dist;
      this.updateCameraRotation();
    }
    if (this.keysPressed.has('ShiftLeft')) {
      const up = this.normalize(this.camera.up);
      this.camera.position[0]-=up[0]*dist; this.camera.position[1]-=up[1]*dist; this.camera.position[2]-=up[2]*dist;
      this.updateCameraRotation();
    }
  }

  private getForward(): [number,number,number] {
    const f: [number,number,number] = [
      this.camera.look_at[0]-this.camera.position[0],
      this.camera.look_at[1]-this.camera.position[1],
      this.camera.look_at[2]-this.camera.position[2],
    ];
    return this.normalize(f);
  }
  private getRight(): [number,number,number] {
    const f = this.getForward();
    const u = this.camera.up;
    const r: [number,number,number] = [
      f[1]*u[2]-f[2]*u[1],
      f[2]*u[0]-f[0]*u[2],
      f[0]*u[1]-f[1]*u[0]
    ];
    return this.normalize(r);
  }
  private normalize(v: [number,number,number]): [number,number,number] {
    const l = Math.hypot(v[0],v[1],v[2]) || 1;
    return [v[0]/l,v[1]/l,v[2]/l];
  }
  private move(dir: [number,number,number], d: number) {
    this.camera.position[0]+=dir[0]*d;
    this.camera.position[1]+=dir[1]*d;
    this.camera.position[2]+=dir[2]*d;
    this.updateCameraRotation();
  }

  setMoveSpeed(s: number) { this.moveSpeed = s; }
  setMouseSensitivity(s: number) { this.mouseSensitivity = s; }
  setInvertY(v: boolean) { this.invertY = v; this.updateCameraRotation(); }

  getDebugInfo() {
    return {
      position: this.camera.position.slice(),
      yaw: this.yaw.toFixed(1),
      pitch: this.pitch.toFixed(1),
      mouseLocked: this.mouseLocked,
      keys: Array.from(this.keysPressed)
    };
  }

  // Public helper to force a lock attempt from console: controls.ensurePointerLock()
  ensurePointerLock() {
    if (!this.mouseLocked) {
      (this.canvas as any).focus?.();
      try {
        const p: any = this.canvas.requestPointerLock();
        if (p && typeof p.catch === 'function') {
          p.catch((err: any)=> console.warn('[Controls] ensurePointerLock rejected:', err));
        }
      } catch (err) {
        console.warn('[Controls] ensurePointerLock threw', err);
      }
    }
  }
}
