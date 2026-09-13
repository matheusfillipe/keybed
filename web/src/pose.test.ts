import { describe, expect, test } from "vitest";
import type { Point } from "./homography";
import {
  canonicalQuad,
  DEPTH_UNITS,
  KEYBED_DEPTH_MM,
  projectPoint,
  solvePose,
  WHITE_KEY_COUNT,
  WHITE_KEY_MM,
} from "./pose";

const WIDTH = 1280;
const HEIGHT = 720;
const CENTER_X = WIDTH / 2;
const CENTER_Y = HEIGHT / 2;
const DEG = Math.PI / 180;
const POSES_PER_FOCAL = 30;
const SAMPLE_ATTEMPTS = 1000;

interface TruePose {
  rotation: number[][];
  translation: number[];
}

const CORNERS_MM: [number, number][] = [
  [0, 0],
  [WHITE_KEY_COUNT * WHITE_KEY_MM, 0],
  [WHITE_KEY_COUNT * WHITE_KEY_MM, KEYBED_DEPTH_MM],
  [0, KEYBED_DEPTH_MM],
];

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rotationX(a: number): number[][] {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rotationY(a: number): number[][] {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

function rotationZ(a: number): number[][] {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

function matMul(a: number[][], b: number[][]): number[][] {
  const out: number[][] = [];
  for (let i = 0; i < 3; i += 1) {
    const row: number[] = [];
    for (let j = 0; j < 3; j += 1) {
      row.push(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]);
    }
    out.push(row);
  }
  return out;
}

function apply(
  r: number[][],
  t: number[],
  p: [number, number, number],
): [number, number, number] {
  return [
    r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0],
    r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1],
    r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2],
  ];
}

function angleBetweenDeg(a: number[], b: number[]): number {
  const na = Math.hypot(a[0], a[1], a[2]);
  const nb = Math.hypot(b[0], b[1], b[2]);
  const cosine = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  return (Math.acos(Math.min(1, Math.max(-1, cosine))) * 180) / Math.PI;
}

function rotationErrorDeg(estimated: number[][], truth: number[][]): number {
  let trace = 0;
  for (let i = 0; i < 3; i += 1) {
    for (let j = 0; j < 3; j += 1) {
      trace += estimated[i][j] * truth[i][j];
    }
  }
  return (
    (Math.acos(Math.min(1, Math.max(-1, (trace - 1) / 2))) * 180) / Math.PI
  );
}

function projectTrue(
  pose: TruePose,
  focal: number,
  xmm: number,
  ymm: number,
): Point {
  const [xc, yc, zc] = apply(pose.rotation, pose.translation, [xmm, ymm, 0]);
  return { x: CENTER_X + (focal * xc) / zc, y: CENTER_Y + (focal * yc) / zc };
}

function samplePose(rand: () => number): TruePose {
  for (let attempt = 0; attempt < SAMPLE_ATTEMPTS; attempt += 1) {
    const pitch = (rand() * 36 - 18) * DEG;
    const yaw = (rand() * 36 - 18) * DEG;
    const roll = (rand() * 16 - 8) * DEG;
    if (Math.hypot(pitch, yaw) < 4 * DEG) {
      continue;
    }
    const rotation = matMul(
      rotationX(pitch),
      matMul(rotationY(yaw), rotationZ(roll)),
    );
    const distance = 500 + rand() * 700;
    const jitterX = (rand() * 2 - 1) * 40;
    const jitterY = (rand() * 2 - 1) * 40;
    const center = apply(
      rotation,
      [0, 0, 0],
      [
        (WHITE_KEY_COUNT * WHITE_KEY_MM) / 2,
        (DEPTH_UNITS * WHITE_KEY_MM) / 2,
        0,
      ],
    );
    const translation = [-center[0] + jitterX, -center[1] + jitterY, distance];
    return { rotation, translation };
  }
  throw new Error("no valid pose sampled");
}

function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

describe("pose", () => {
  test("synthetic camera round trip", () => {
    const rand = mulberry32(42);
    const interiorUs = [13, 26, 39];
    const interiorVs = [0.25, 0.5, 0.75];
    for (const focal of [600, 800, 1200]) {
      for (let i = 0; i < POSES_PER_FOCAL; i += 1) {
        const truth = samplePose(rand);
        const imageCorners = CORNERS_MM.map(([xmm, ymm]) =>
          projectTrue(truth, focal, xmm, ymm),
        );
        const pose = solvePose(imageCorners, WIDTH, HEIGHT);
        expect(Math.abs(pose.focal - focal) / focal).toBeLessThan(0.05);
        expect(rotationErrorDeg(pose.rotation, truth.rotation)).toBeLessThan(
          1.5,
        );
        expect(
          angleBetweenDeg(pose.translation, truth.translation),
        ).toBeLessThan(5);
        for (const [j, [xmm, ymm]] of CORNERS_MM.entries()) {
          const projected = projectPoint(
            pose,
            xmm / WHITE_KEY_MM,
            ymm / WHITE_KEY_MM,
            WIDTH,
            HEIGHT,
          );
          expect(distance(projected, imageCorners[j])).toBeLessThan(1.5);
        }
        for (const u of interiorUs) {
          for (const v of interiorVs) {
            const projected = projectPoint(
              pose,
              u,
              v * DEPTH_UNITS,
              WIDTH,
              HEIGHT,
            );
            const truthPoint = projectTrue(
              truth,
              focal,
              u * WHITE_KEY_MM,
              v * DEPTH_UNITS * WHITE_KEY_MM,
            );
            expect(distance(projected, truthPoint)).toBeLessThan(2);
          }
        }
      }
    }
  });
});

describe("canonicalQuad", () => {
  const depthFirst: Point[] = [
    { x: 254, y: 5 },
    { x: 314, y: 5 },
    { x: 298, y: 472 },
    { x: 185, y: 469 },
  ];

  test("puts the key span on the first edge", () => {
    const quad = canonicalQuad(depthFirst);
    const span = Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y);
    const depth = Math.hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y);
    expect(span).toBeGreaterThan(depth);
  });

  test("keeps the same four points", () => {
    const key = (p: Point): string => `${p.x},${p.y}`;
    expect(canonicalQuad(depthFirst).map(key).sort()).toEqual(
      depthFirst.map(key).sort(),
    );
  });

  test("is idempotent", () => {
    const once = canonicalQuad(depthFirst);
    expect(canonicalQuad(once)).toEqual(once);
  });
});
