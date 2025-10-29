export class Camera {
  position: [number, number, number];
  look_at: [number, number, number];
  up: [number, number, number];

  constructor(
    position: [number, number, number] = [0,0,3],
    lookAt: [number, number, number] = [0,0,0],
    up: [number, number, number] = [0,1,0]
  ) {
    this.position = [...position];
    this.look_at = [...lookAt];
    this.up = [...up];
  }

  setPosition(x: number, y: number, z: number) {
    this.position[0]=x; this.position[1]=y; this.position[2]=z;
  }

  setLookAt(x: number, y: number, z: number) {
    this.look_at[0]=x; this.look_at[1]=y; this.look_at[2]=z;
  }

  setUp(x: number, y: number, z: number) {
    this.up[0]=x; this.up[1]=y; this.up[2]=z;
  }
}
