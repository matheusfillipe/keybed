import { describe, expect, test } from "vitest";
import type { Point } from "./homography";
import { createSteady } from "./steady";

const KEYBED: Point[] = [
  { x: 0.1, y: 0.4 },
  { x: 0.9, y: 0.4 },
  { x: 0.92, y: 0.55 },
  { x: 0.08, y: 0.55 },
];

function shifted(quad: Point[], dx: number, dy: number): Point[] {
  return quad.map((p) => ({ x: p.x + dx, y: p.y + dy }));
}

function farthest(a: Point[], b: Point[]): number {
  return Math.max(...a.map((p, i) => Math.hypot(p.x - b[i].x, p.y - b[i].y)));
}

describe("steady", () => {
  test("a still keybed with sub-pixel noise settles onto the truth", () => {
    const steady = createSteady();
    let out = steady.accept(KEYBED);
    for (let i = 0; i < 20; i += 1) {
      const noise = (i % 2 === 0 ? 1 : -1) * 0.001;
      out = steady.accept(shifted(KEYBED, noise, -noise));
    }
    expect(farthest(out, KEYBED)).toBeLessThan(0.0015);
  });

  test("while still, refit noise is cut to under half", () => {
    const steady = createSteady();
    steady.accept(KEYBED, true);
    let out = KEYBED;
    for (let i = 0; i < 20; i += 1) {
      out = steady.accept(shifted(KEYBED, i % 2 ? 0.004 : -0.004, 0), true);
    }
    expect(farthest(out, KEYBED)).toBeLessThan(0.002);
  });

  test("while still, a correction is drawn within two seconds, not crawled to", () => {
    const steady = createSteady();
    steady.accept(KEYBED, true);
    const corrected = shifted(KEYBED, 0.02, 0.01);
    let out = KEYBED;
    for (let i = 0; i < 8; i += 1) {
      out = steady.accept(corrected, true);
    }
    expect(farthest(out, corrected)).toBeLessThan(0.003);
  });

  test("a real move is followed, not lagged away", () => {
    const steady = createSteady();
    steady.accept(KEYBED);
    const moved = shifted(KEYBED, 0.02, 0.01);
    let out = moved;
    for (let i = 0; i < 12; i += 1) {
      out = steady.accept(moved);
    }
    expect(farthest(out, moved)).toBeLessThan(0.003);
  });

  test("one wild frame is held through, a run of them is followed", () => {
    const steady = createSteady();
    steady.accept(KEYBED);
    const wild = shifted(KEYBED, 0.3, 0);
    expect(farthest(steady.accept(wild), KEYBED)).toBeLessThan(0.001);
    let out = KEYBED;
    for (let i = 0; i < 8; i += 1) {
      out = steady.accept(wild);
    }
    expect(farthest(out, wild)).toBe(0);
  });
});
