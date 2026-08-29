import * as THREE from "three";
import { createRoll } from "./roll";

const KEYBOARD_WIDTH = 52;
const KEYBOARD_DEPTH = 6.5;
export const DEFAULT_ROLL_TILT_DEG = 20;
export const DEFAULT_ROLL_HEIGHT = 26;

export interface Scene3D {
  root: THREE.Scene;
  resize(width: number, height: number): void;
  render(nowMs: number): void;
  setRollTilt(degrees: number): void;
  setRollHeight(units: number): void;
}

export function createScene(stripCanvas: HTMLCanvasElement): Scene3D {
  const canvas = document.createElement("canvas");
  canvas.style.position = "fixed";
  canvas.style.inset = "0";
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
  document.body.appendChild(canvas);

  const renderer = new THREE.WebGLRenderer({
    canvas,
    alpha: true,
    antialias: true,
  });
  renderer.setClearColor(0x000000, 0);

  const root = new THREE.Scene();
  root.add(new THREE.AmbientLight(0xffffff, 0.35));

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 300);
  camera.position.set(0, 9, 40);
  camera.lookAt(0, 6, 0);

  const keyboardTexture = new THREE.CanvasTexture(stripCanvas);
  keyboardTexture.colorSpace = THREE.SRGBColorSpace;
  const keyboard = new THREE.Mesh(
    new THREE.PlaneGeometry(KEYBOARD_WIDTH, KEYBOARD_DEPTH),
    new THREE.MeshLambertMaterial({
      map: keyboardTexture,
      emissive: 0x8c8c8c,
      emissiveMap: keyboardTexture,
    }),
  );
  keyboard.rotation.x = -Math.PI / 2;
  root.add(keyboard);

  const roll = createRoll(root);
  const rollTexture = new THREE.CanvasTexture(roll.canvas);
  rollTexture.colorSpace = THREE.SRGBColorSpace;
  const rollGeometry = new THREE.PlaneGeometry(KEYBOARD_WIDTH, 1);
  rollGeometry.translate(0, 0.5, 0);
  const rollMesh = new THREE.Mesh(
    rollGeometry,
    new THREE.MeshBasicMaterial({
      map: rollTexture,
      transparent: true,
      side: THREE.DoubleSide,
    }),
  );
  rollMesh.position.set(0, 0, -KEYBOARD_DEPTH / 2);
  rollMesh.rotation.x = -THREE.MathUtils.degToRad(DEFAULT_ROLL_TILT_DEG);
  rollMesh.scale.y = DEFAULT_ROLL_HEIGHT;
  root.add(rollMesh);

  const resize = (width: number, height: number): void => {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  resize(window.innerWidth, window.innerHeight);

  return {
    root,
    resize,
    render: (nowMs) => {
      roll.update(nowMs);
      keyboardTexture.needsUpdate = true;
      rollTexture.needsUpdate = true;
      renderer.render(root, camera);
    },
    setRollTilt: (degrees) => {
      rollMesh.rotation.x = -THREE.MathUtils.degToRad(degrees);
    },
    setRollHeight: (units) => {
      rollMesh.scale.y = units;
    },
  };
}
