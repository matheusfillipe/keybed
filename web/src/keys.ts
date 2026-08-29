const NOTE_NAMES = [
  "C",
  "C#",
  "D",
  "D#",
  "E",
  "F",
  "F#",
  "G",
  "G#",
  "A",
  "A#",
  "B",
];
const WHITE_PC_INDEX = [0, -1, 1, -1, 2, 3, -1, 4, -1, 5, -1, 6];
const BLACK_OFFSETS = [-1, 0.6, -1, 1.75, -1, -1, 3.6, -1, 4.63, -1, 5.66, -1];
const BLACK_WIDTH = 0.58;
const WHITE_COUNT = 52;
export const LOW_PITCH = 21;
export const HIGH_PITCH = 108;

export interface KeyRect {
  u0: number;
  u1: number;
}

const WHITE_PITCHES: number[] = [];
for (let pitch = LOW_PITCH; pitch <= HIGH_PITCH; pitch += 1) {
  if (!isBlack(pitch)) {
    WHITE_PITCHES.push(pitch);
  }
}

export const KEY_NAMES: string[] = Array.from(
  { length: HIGH_PITCH - LOW_PITCH + 1 },
  (_, i) => {
    const pitch = LOW_PITCH + i;
    return NOTE_NAMES[pitch % 12] + String(Math.floor(pitch / 12) - 1);
  },
);

export function isBlack(pitch: number): boolean {
  return BLACK_OFFSETS[pitch % 12] >= 0;
}

export function whiteIndex(pitch: number): number {
  return 7 * Math.floor(pitch / 12) - 12 + WHITE_PC_INDEX[pitch % 12];
}

export function keyRect(pitch: number): KeyRect {
  const offset = BLACK_OFFSETS[pitch % 12];
  if (offset >= 0) {
    const base = 7 * Math.floor(pitch / 12) - 12;
    const u0 = (base + offset) / WHITE_COUNT;
    return { u0, u1: u0 + BLACK_WIDTH / WHITE_COUNT };
  }
  const index = whiteIndex(pitch);
  return { u0: index / WHITE_COUNT, u1: (index + 1) / WHITE_COUNT };
}

export function pitchAt(u: number): number | null {
  if (u < 0 || u > 1) {
    return null;
  }
  const w = u * WHITE_COUNT;
  for (let pitch = LOW_PITCH; pitch <= HIGH_PITCH; pitch += 1) {
    if (!isBlack(pitch)) {
      continue;
    }
    const rect = keyRect(pitch);
    if (w >= rect.u0 * WHITE_COUNT && w < rect.u1 * WHITE_COUNT) {
      return pitch;
    }
  }
  const index = Math.min(WHITE_COUNT - 1, Math.max(0, Math.floor(w)));
  return WHITE_PITCHES[index];
}
