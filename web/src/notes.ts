import type { Scene } from "three";
import { AdditiveBlending, BoxGeometry, Mesh, MeshBasicMaterial } from "three";
import { isBlack, keyRect } from "./keys";
import { KEYBED_DEPTH_MM, WHITE_KEY_COUNT, WHITE_KEY_MM } from "./pose";

const KEYBED_WIDTH_MM = WHITE_KEY_COUNT * WHITE_KEY_MM;
const NOTE_HEIGHT_MM = 240;
const NOTE_DEPTH_MM = 30;
const NOTE_GAP_MM = 2;
const RISE_MM = 600;
const LIFE_MS = 1200;
const SPAWN_INTERVAL_MS = 300;
const FADE_IN = 0.12;
const FADE_OUT = 0.25;
const BASE_PITCH = 48;
const OCTAVES = 3;
const PENTATONIC = [0, 2, 4, 7, 9];
const COLORS = { white: 0x00ffff, black: 0xff00ff };

type Tone = keyof typeof COLORS;

interface ActiveNote {
  mesh: Mesh;
  material: MeshBasicMaterial;
  tone: Tone;
  width: number;
  bornAt: number;
}

export interface NoteField {
  setEnabled(enabled: boolean): void;
  update(nowMs: number): void;
}

const PITCHES: number[] = [];
for (let octave = 0; octave < OCTAVES; octave += 1) {
  for (const degree of PENTATONIC) {
    PITCHES.push(BASE_PITCH + 12 * octave + degree);
  }
}

function toneOf(pitch: number): Tone {
  return isBlack(pitch) ? "black" : "white";
}

function keyWidthMm(pitch: number): number {
  const rect = keyRect(pitch);
  return (rect.u1 - rect.u0) * KEYBED_WIDTH_MM - NOTE_GAP_MM;
}

function createMaterial(color: number): MeshBasicMaterial {
  return new MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0,
    blending: AdditiveBlending,
    depthWrite: false,
  });
}

export function createNotes(scene: Scene): NoteField {
  const geometry = new BoxGeometry(1, NOTE_DEPTH_MM, NOTE_HEIGHT_MM);
  const pools: Record<Tone, MeshBasicMaterial[]> = { white: [], black: [] };
  const active: ActiveNote[] = [];
  let interval: number | null = null;

  const acquire = (tone: Tone): MeshBasicMaterial => {
    const pooled = pools[tone].pop();
    if (pooled) {
      pooled.opacity = 0;
      return pooled;
    }
    return createMaterial(COLORS[tone]);
  };

  const release = (note: ActiveNote): void => {
    scene.remove(note.mesh);
    pools[note.tone].push(note.material);
  };

  const clear = (): void => {
    for (const note of active) {
      release(note);
    }
    active.length = 0;
  };

  const spawn = (): void => {
    const count = 1 + Math.floor(Math.random() * 3);
    for (let i = 0; i < count; i += 1) {
      const pitch = PITCHES[Math.floor(Math.random() * PITCHES.length)];
      const tone = toneOf(pitch);
      const material = acquire(tone);
      const mesh = new Mesh(geometry, material);
      const rect = keyRect(pitch);
      mesh.position.set(
        ((rect.u0 + rect.u1) / 2) * KEYBED_WIDTH_MM - KEYBED_WIDTH_MM / 2,
        KEYBED_DEPTH_MM / 2,
        NOTE_HEIGHT_MM / 2,
      );
      const width = keyWidthMm(pitch);
      mesh.scale.set(width, 1, 1);
      scene.add(mesh);
      active.push({
        mesh,
        material,
        tone,
        width,
        bornAt: performance.now(),
      });
    }
  };

  const setEnabled = (enabled: boolean): void => {
    if (enabled) {
      if (interval === null) {
        spawn();
        interval = window.setInterval(spawn, SPAWN_INTERVAL_MS);
      }
      return;
    }
    if (interval !== null) {
      window.clearInterval(interval);
      interval = null;
    }
    clear();
  };

  const update = (nowMs: number): void => {
    for (let i = active.length - 1; i >= 0; i -= 1) {
      const note = active[i];
      const t = (nowMs - note.bornAt) / LIFE_MS;
      if (t >= 1) {
        active.splice(i, 1);
        release(note);
        continue;
      }
      note.mesh.position.z = NOTE_HEIGHT_MM / 2 + RISE_MM * (1 - (1 - t) ** 3);
      const fadeIn = Math.min(1, t / FADE_IN);
      const fadeOut = t < 1 - FADE_OUT ? 0 : (t - (1 - FADE_OUT)) / FADE_OUT;
      note.material.opacity = fadeIn * (1 - fadeOut);
      const shrink = 1 - fadeOut;
      note.mesh.scale.set(note.width * shrink, shrink, 1);
    }
  };

  return { setEnabled, update };
}
