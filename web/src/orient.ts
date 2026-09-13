import { applyHomography, findHomography, type Point } from "./homography";
import { canonicalQuad } from "./pose";

const UNIT_SQUARE: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];
const COLUMNS = 32;
const ROWS = 5;
const BACK_BAND = [0.06, 0.32];
const FRONT_BAND = [0.68, 0.94];
const MIN_SAMPLES = 20;

// measured over the whole corpus: a quad sitting on the keybed never scored below 0.105
export const ON_KEYBED_MARGIN = 0.1;

export interface Facing {
  quad: Point[];
  margin: number;
  onKeybed: boolean;
}

function bandMean(
  gray: Float32Array,
  size: number,
  quad: Point[],
  band: number[],
): number | null {
  const h = findHomography(UNIT_SQUARE, quad);
  let total = 0;
  let count = 0;
  for (let row = 0; row < ROWS; row += 1) {
    const v = band[0] + ((band[1] - band[0]) * row) / (ROWS - 1);
    for (let column = 0; column < COLUMNS; column += 1) {
      const u = 0.04 + (0.92 * column) / (COLUMNS - 1);
      const p = applyHomography(h, u, v);
      const x = Math.round(p.x * size);
      const y = Math.round(p.y * size);
      if (x < 0 || y < 0 || x >= size || y >= size) {
        continue;
      }
      total += gray[y * size + x];
      count += 1;
    }
  }
  return count < MIN_SAMPLES ? null : total / count;
}

// The corner order already carries direction: the net is trained on back-edge-first labels and
// scored a positive margin on 110 of 110 quads it placed within 30 px. Rotating on a negative
// margin only ever fired on quads that were already wrong, so this reports instead of correcting.
export function facing(
  gray: Float32Array,
  size: number,
  quad: Point[],
): Facing {
  const ordered = canonicalQuad(quad);
  const back = bandMean(gray, size, ordered, BACK_BAND);
  const front = bandMean(gray, size, ordered, FRONT_BAND);
  if (back === null || front === null) {
    return { quad: ordered, margin: 0, onKeybed: false };
  }
  const margin = (front - back) / (front + back + 1e-6);
  return { quad: ordered, margin, onKeybed: margin >= ON_KEYBED_MARGIN };
}
