import * as THREE from "three";
import { isBlack, keyRect } from "./keys";

const ROLL_WIDTH = 1280;
const ROLL_HEIGHT = 640;
const FALL_SECONDS = 2.5;
const SPAWN_INTERVAL_MS = 300;
const MAX_NOTES_PER_SPAWN = 3;
const MIN_DURATION_MS = 150;
const MAX_DURATION_MS = 500;
const FLASH_MS = 300;
const FLASH_ALPHA = 0.55;
const FLASH_LIFT = 0.02;
const KEYBOARD_WIDTH = 52;
const KEYBOARD_DEPTH = 6.5;
const SCALE_PITCHES = [60, 62, 64, 67, 69, 72, 74, 76, 79, 81, 84];
const WHITE_COLOR = "#38bdf8";
const BLACK_COLOR = "#f472b6";

interface Note {
  pitch: number;
  startMs: number;
  durationMs: number;
  flashed: boolean;
}

interface Flash {
  mesh: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  startMs: number;
}

export interface Roll {
  canvas: HTMLCanvasElement;
  update(nowMs: number): void;
}

export function createRoll(parent: THREE.Object3D): Roll {
  const canvas = document.createElement("canvas");
  canvas.width = ROLL_WIDTH;
  canvas.height = ROLL_HEIGHT;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2d canvas context unavailable");
  }

  const notes: Note[] = [];
  const flashes: Flash[] = [];
  let lastSpawnMs: number | null = null;

  const pitchColor = (pitch: number): string =>
    isBlack(pitch) ? BLACK_COLOR : WHITE_COLOR;

  const spawn = (nowMs: number): void => {
    const count = 1 + Math.floor(Math.random() * MAX_NOTES_PER_SPAWN);
    for (let i = 0; i < count; i += 1) {
      const pitch =
        SCALE_PITCHES[Math.floor(Math.random() * SCALE_PITCHES.length)];
      const durationMs =
        MIN_DURATION_MS + Math.random() * (MAX_DURATION_MS - MIN_DURATION_MS);
      notes.push({ pitch, startMs: nowMs, durationMs, flashed: false });
    }
  };

  const drawNote = (note: Note, progress: number): void => {
    const rect = keyRect(note.pitch);
    const x = rect.u0 * ROLL_WIDTH;
    const w = (rect.u1 - rect.u0) * ROLL_WIDTH;
    const bottom = progress * ROLL_HEIGHT;
    const top =
      bottom - (note.durationMs / (FALL_SECONDS * 1000)) * ROLL_HEIGHT;
    if (bottom <= 0 || top >= ROLL_HEIGHT) {
      return;
    }
    ctx.fillStyle = pitchColor(note.pitch);
    ctx.beginPath();
    ctx.roundRect(x, top, w, bottom - top, w * 0.3);
    ctx.fill();
  };

  const startFlash = (note: Note, nowMs: number): void => {
    const rect = keyRect(note.pitch);
    const x0 = rect.u0 * KEYBOARD_WIDTH - KEYBOARD_WIDTH / 2;
    const w = (rect.u1 - rect.u0) * KEYBOARD_WIDTH;
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(w, KEYBOARD_DEPTH),
      new THREE.MeshBasicMaterial({
        color: pitchColor(note.pitch),
        transparent: true,
        opacity: FLASH_ALPHA,
        depthWrite: false,
      }),
    );
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(x0 + w / 2, FLASH_LIFT, 0);
    parent.add(mesh);
    flashes.push({ mesh, startMs: nowMs });
  };

  return {
    canvas,
    update: (nowMs) => {
      if (lastSpawnMs === null || nowMs - lastSpawnMs >= SPAWN_INTERVAL_MS) {
        lastSpawnMs = nowMs;
        spawn(nowMs);
      }

      ctx.clearRect(0, 0, ROLL_WIDTH, ROLL_HEIGHT);
      ctx.fillStyle = "rgba(8, 12, 18, 0.55)";
      ctx.fillRect(0, 0, ROLL_WIDTH, ROLL_HEIGHT);

      for (let i = notes.length - 1; i >= 0; i -= 1) {
        const note = notes[i];
        const progress = (nowMs - note.startMs) / (FALL_SECONDS * 1000);
        const tailProgress = progress - note.durationMs / (FALL_SECONDS * 1000);
        if (tailProgress > 1) {
          notes.splice(i, 1);
          continue;
        }
        drawNote(note, progress);
        if (!note.flashed && progress >= 1) {
          note.flashed = true;
          startFlash(note, nowMs);
        }
      }

      for (let i = flashes.length - 1; i >= 0; i -= 1) {
        const flash = flashes[i];
        const elapsed = nowMs - flash.startMs;
        if (elapsed >= FLASH_MS) {
          parent.remove(flash.mesh);
          flash.mesh.geometry.dispose();
          flash.mesh.material.dispose();
          flashes.splice(i, 1);
          continue;
        }
        flash.mesh.material.opacity = FLASH_ALPHA * (1 - elapsed / FLASH_MS);
      }
    },
  };
}
