import type { Point } from "./homography";

// the keybed does not move, so a corner that jumps this far in one frame is a miss, not motion
const JUMP_LIMIT = 0.08;
// enough consecutive jumps means the keyboard or the camera really did move, so follow it
const JUMPS_BEFORE_RESET = 8;
// One-Euro (Casiez 2012): the cutoff rises with speed, so a still keybed is smoothed hard and a
// camera pan is followed without lag. Units are frame fractions per second.
const DETECT_HZ = 4;
const MIN_CUTOFF_HZ = 1.0;
// while the camera is still, a pixel or two of refit noise is smoothed away since the fit
// is already steady; at 0.05 Hz a real correction crawled into place over ten seconds instead
const STILL_CUTOFF_HZ = 0.5;
// a corner moving a fiftieth of the frame per second is a correction, not noise, and doubles
// the cutoff; a pan of half a frame per second is followed at once
const BETA = 20;
const DERIVATIVE_CUTOFF_HZ = 1.0;

function alpha(cutoffHz: number): number {
  return 1 / (1 + DETECT_HZ / (2 * Math.PI * cutoffHz));
}

export interface Steady {
  accept(quad: Point[], still?: boolean): Point[];
  reset(): void;
}

function farthestCorner(a: Point[], b: Point[]): number {
  let worst = 0;
  for (let i = 0; i < 4; i += 1) {
    worst = Math.max(worst, Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y));
  }
  return worst;
}

export function createSteady(): Steady {
  let held: Point[] | null = null;
  let rate: Point[] = [];
  let jumps = 0;
  return {
    accept(quad, still = false) {
      if (!held) {
        held = quad;
        return held;
      }
      if (farthestCorner(quad, held) > JUMP_LIMIT) {
        jumps += 1;
        if (jumps < JUMPS_BEFORE_RESET) {
          return held;
        }
        held = quad;
        jumps = 0;
        return held;
      }
      jumps = 0;
      const previous = held;
      if (rate.length !== 4) {
        rate = quad.map(() => ({ x: 0, y: 0 }));
      }
      const aRate = alpha(DERIVATIVE_CUTOFF_HZ);
      held = quad.map((p, i) => {
        const dx = (p.x - previous[i].x) * DETECT_HZ;
        const dy = (p.y - previous[i].y) * DETECT_HZ;
        rate[i] = {
          x: aRate * dx + (1 - aRate) * rate[i].x,
          y: aRate * dy + (1 - aRate) * rate[i].y,
        };
        const floor = still ? STILL_CUTOFF_HZ : MIN_CUTOFF_HZ;
        const ax = alpha(floor + BETA * Math.abs(rate[i].x));
        const ay = alpha(floor + BETA * Math.abs(rate[i].y));
        return {
          x: previous[i].x + (p.x - previous[i].x) * ax,
          y: previous[i].y + (p.y - previous[i].y) * ay,
        };
      });
      return held;
    },
    reset() {
      held = null;
      rate = [];
      jumps = 0;
    },
  };
}
