import type { Point } from "./homography";

const MIN_ASPECT = 1.8;
const MAX_ASPECT = 30.0;
const MIN_SHORT_EDGE = 0.01;
// one end may be drawn this much wider than the other, the same as their distance ratio; the
// hand-labelled views span 1.54 to 2.65
const MAX_END_RATIO = 3.2;

export interface QuadCheck {
  usable: boolean;
  reason: string;
}

function cross(a: Point, b: Point, c: Point): number {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function edge(quad: readonly Point[], i: number): number {
  const a = quad[i];
  const b = quad[(i + 1) % 4];
  return Math.hypot(b.x - a.x, b.y - a.y);
}

// a label saved from a collapsed or folded quad is worse than no label, so capture refuses it
export function checkQuad(quad: readonly Point[]): QuadCheck {
  if (quad.length !== 4) {
    return { usable: false, reason: "needs 4 corners" };
  }
  const signs = [0, 1, 2, 3].map((i) =>
    Math.sign(cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4])),
  );
  if (signs.some((s) => s === 0) || new Set(signs).size !== 1) {
    return { usable: false, reason: "corners fold over" };
  }
  const edges = [0, 1, 2, 3].map((i) => edge(quad, i));
  const short = Math.min(...edges);
  if (short < MIN_SHORT_EDGE) {
    return { usable: false, reason: "corners collapsed" };
  }
  const span = (edges[0] + edges[2]) / 2;
  const depth = (edges[1] + edges[3]) / 2;
  // the 52 white keys always run along edge 0->1, so a keybed is longer than it is deep;
  // an unsigned ratio would accept a quad lying across the keys and read it as a good fit
  const aspect = span / Math.max(depth, MIN_SHORT_EDGE);
  if (aspect < MIN_ASPECT || aspect > MAX_ASPECT) {
    return {
      usable: false,
      reason: `aspect ${aspect.toFixed(1)} is not a keybed`,
    };
  }
  const ends = [edges[1], edges[3]];
  const ratio = Math.max(...ends) / Math.max(Math.min(...ends), MIN_SHORT_EDGE);
  if (ratio > MAX_END_RATIO) {
    return {
      usable: false,
      reason: `one end drawn ${ratio.toFixed(1)}x the other`,
    };
  }
  return { usable: true, reason: "" };
}
