import type { PerspectiveCamera } from "three";
import { Vector3 } from "three";
import type { Point } from "./homography";

// Measured from piano_keys.glb: white keys top out at y 0.610, black keys sit on the -x half,
// so -x is the back edge. Span 24.83 by depth 3.058 is aspect 8.12, against 8.15 for a real keybed.
export const KEY_TOP_Y = 0.61;
export const BACK_X = -1.627;
export const FRONT_X = 1.431;
export const SPAN_MIN_Z = -16.399;
export const SPAN_MAX_Z = 8.431;

// corner 0 to 1 runs along the back edge, 1 to 2 crosses the depth, matching the label convention
export const KEYBED_CORNERS: Vector3[] = [
  new Vector3(BACK_X, KEY_TOP_Y, SPAN_MIN_Z),
  new Vector3(BACK_X, KEY_TOP_Y, SPAN_MAX_Z),
  new Vector3(FRONT_X, KEY_TOP_Y, SPAN_MAX_Z),
  new Vector3(FRONT_X, KEY_TOP_Y, SPAN_MIN_Z),
];

export const KEYBED_CENTRE = new Vector3(
  (BACK_X + FRONT_X) / 2,
  KEY_TOP_Y,
  (SPAN_MIN_Z + SPAN_MAX_Z) / 2,
);

export const SPAN = SPAN_MAX_Z - SPAN_MIN_Z;
export const DEPTH = FRONT_X - BACK_X;

const scratch = new Vector3();

export function projectCorners(camera: PerspectiveCamera): Point[] {
  return KEYBED_CORNERS.map((corner) => {
    scratch.copy(corner).project(camera);
    return { x: (scratch.x + 1) / 2, y: (1 - scratch.y) / 2 };
  });
}

const GRID = 9;
const sampleA = new Vector3();
const sampleB = new Vector3();
const sample = new Vector3();

// most real frames cut the keybed at the image edge, so what matters is how much of it landed,
// not whether all four corners did
export function visibleFraction(camera: PerspectiveCamera): number {
  let inside = 0;
  for (let i = 0; i < GRID; i += 1) {
    const u = i / (GRID - 1);
    sampleA.lerpVectors(KEYBED_CORNERS[0], KEYBED_CORNERS[1], u);
    sampleB.lerpVectors(KEYBED_CORNERS[3], KEYBED_CORNERS[2], u);
    for (let j = 0; j < GRID; j += 1) {
      sample.lerpVectors(sampleA, sampleB, j / (GRID - 1)).project(camera);
      if (
        sample.z < 1 &&
        sample.x >= -1 &&
        sample.x <= 1 &&
        sample.y >= -1 &&
        sample.y <= 1
      ) {
        inside += 1;
      }
    }
  }
  return inside / (GRID * GRID);
}
