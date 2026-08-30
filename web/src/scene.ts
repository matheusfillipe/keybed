import { Matrix4, PerspectiveCamera, Scene, WebGLRenderer } from "three";
import type { PlanePose } from "./pose";

const NEAR_MM = 1;
const FAR_MM = 20000;
const cameraBasis = new Matrix4();

export interface OverlayScene {
  readonly scene: Scene;
  setPose(pose: PlanePose, width: number, height: number): void;
  render(): void;
}

export function applyPose(
  camera: PerspectiveCamera,
  pose: PlanePose,
  width: number,
  height: number,
): void {
  camera.fov = (2 * Math.atan(height / (2 * pose.focal)) * 180) / Math.PI;
  camera.aspect = width / height;
  const r = pose.rotation;
  const t = pose.translation;
  camera.position.set(
    -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
    -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
    -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
  );
  cameraBasis.set(
    r[0][0],
    -r[1][0],
    -r[2][0],
    0,
    r[0][1],
    -r[1][1],
    -r[2][1],
    0,
    r[0][2],
    -r[1][2],
    -r[2][2],
    0,
    0,
    0,
    0,
    1,
  );
  camera.quaternion.setFromRotationMatrix(cameraBasis);
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
}

export function createOverlayScene(): OverlayScene {
  const canvas = document.createElement("canvas");
  canvas.style.position = "fixed";
  canvas.style.inset = "0";
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
  canvas.style.pointerEvents = "none";
  document.body.appendChild(canvas);
  const renderer = new WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setClearAlpha(0);
  const scene = new Scene();
  const camera = new PerspectiveCamera(60, 1, NEAR_MM, FAR_MM);
  let sizedWidth = 0;
  let sizedHeight = 0;
  return {
    scene,
    setPose: (pose, width, height) => {
      if (width !== sizedWidth || height !== sizedHeight) {
        sizedWidth = width;
        sizedHeight = height;
        renderer.setSize(width, height, false);
      }
      applyPose(camera, pose, width, height);
    },
    render: () => {
      renderer.render(scene, camera);
    },
  };
}
