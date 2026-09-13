import { describe, expect, test } from "vitest";
import { quadFromMask } from "./fitquad";
import type { Point } from "./homography";

const WIDTH = 160;
const HEIGHT = 120;
const QUAD: Point[] = [
  { x: 20, y: 40 },
  { x: 140, y: 30 },
  { x: 144, y: 58 },
  { x: 16, y: 70 },
];

function inside(quad: Point[], x: number, y: number): boolean {
  let hit = false;
  for (let i = 0, j = quad.length - 1; i < quad.length; j = i, i += 1) {
    const a = quad[i];
    const b = quad[j];
    if (
      a.y > y !== b.y > y &&
      x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x
    ) {
      hit = !hit;
    }
  }
  return hit;
}

function rasterise(quad: Point[]): Uint8Array {
  const mask = new Uint8Array(WIDTH * HEIGHT);
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      mask[y * WIDTH + x] = inside(quad, x + 0.5, y + 0.5) ? 1 : 0;
    }
  }
  return mask;
}

function match(pred: Point[], truth: Point[]): number {
  let best = Infinity;
  for (const reversed of [false, true]) {
    const ordered = reversed ? [...pred].reverse() : pred;
    for (let roll = 0; roll < 4; roll += 1) {
      let sum = 0;
      for (let i = 0; i < 4; i += 1) {
        const p = ordered[(i + roll) % 4];
        sum += Math.hypot(p.x - truth[i].x, p.y - truth[i].y);
      }
      best = Math.min(best, sum / 4);
    }
  }
  return best;
}

describe("quadFromMask", () => {
  test("recovers a clean quad to about a pixel", () => {
    const found = quadFromMask(rasterise(QUAD), WIDTH, HEIGHT);
    expect(found).not.toBeNull();
    expect(match(found as Point[], QUAD)).toBeLessThan(2);
  });

  test("ignores a smaller blob elsewhere", () => {
    const mask = rasterise(QUAD);
    for (let y = 100; y < 112; y += 1) {
      for (let x = 4; x < 16; x += 1) {
        mask[y * WIDTH + x] = 1;
      }
    }
    const found = quadFromMask(mask, WIDTH, HEIGHT);
    expect(found).not.toBeNull();
    expect(match(found as Point[], QUAD)).toBeLessThan(3);
  });

  test("returns null on an empty mask", () => {
    expect(
      quadFromMask(new Uint8Array(WIDTH * HEIGHT), WIDTH, HEIGHT),
    ).toBeNull();
  });

  test("returns null on a speck", () => {
    const mask = new Uint8Array(WIDTH * HEIGHT);
    for (let y = 60; y < 63; y += 1) {
      for (let x = 60; x < 63; x += 1) {
        mask[y * WIDTH + x] = 1;
      }
    }
    expect(quadFromMask(mask, WIDTH, HEIGHT)).toBeNull();
  });
});
