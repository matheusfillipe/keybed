const WHITE_PC_INDEX = [0, -1, 1, -1, 2, 3, -1, 4, -1, 5, -1, 6];
const BLACK_OFFSETS = [-1, 0.6, -1, 1.75, -1, -1, 3.6, -1, 4.63, -1, 5.66, -1];
const BLACK_WIDTH = 0.58;
// the instrument in front of the camera: a 61-key board, C2 to C7
export const LOW_PITCH = 36;
export const HIGH_PITCH = 96;

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
export const WHITE_COUNT = WHITE_PITCHES.length;
const FIRST_WHITE = whiteIndex(LOW_PITCH);

export function isBlack(pitch: number): boolean {
  return BLACK_OFFSETS[pitch % 12] >= 0;
}

export function whiteIndex(pitch: number): number {
  return 7 * Math.floor(pitch / 12) - 12 + WHITE_PC_INDEX[pitch % 12];
}

export function keyRect(pitch: number): KeyRect {
  const offset = BLACK_OFFSETS[pitch % 12];
  if (offset >= 0) {
    const base = 7 * Math.floor(pitch / 12) - 12 - FIRST_WHITE;
    const u0 = (base + offset) / WHITE_COUNT;
    return { u0, u1: u0 + BLACK_WIDTH / WHITE_COUNT };
  }
  const index = whiteIndex(pitch) - FIRST_WHITE;
  return { u0: index / WHITE_COUNT, u1: (index + 1) / WHITE_COUNT };
}
