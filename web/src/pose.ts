import { findHomography, type Point } from "./homography";
import { HIGH_PITCH, isBlack, keyRect, LOW_PITCH } from "./keys";

export const WHITE_KEY_MM = 23.5;
export const KEYBED_DEPTH_MM = 150;
export const WHITE_KEY_COUNT = 52;
export const DEPTH_UNITS = KEYBED_DEPTH_MM / WHITE_KEY_MM;
const BLACK_DEPTH_RATIO = 0.62;
const SCAN_SAMPLES = 200;
const GOLDEN_ITERATIONS = 100;
const GOLDEN_RATIO = (Math.sqrt(5) - 1) / 2;

export interface PlanePose {
  focal: number;
  rotation: number[][];
  translation: number[];
  worldWidthMm: number;
  residual: number;
}

export interface GridGeometry {
  separators: [Point, Point][];
  blackQuads: Point[][];
}

const WORLD_CORNERS: Point[] = [
  { x: 0, y: 0 },
  { x: WHITE_KEY_COUNT, y: 0 },
  { x: WHITE_KEY_COUNT, y: DEPTH_UNITS },
  { x: 0, y: DEPTH_UNITS },
];

function inverseCalibrationHomography(
  focal: number,
  width: number,
  height: number,
  h: number[],
): number[][] {
  const cx = width / 2;
  const cy = height / 2;
  const b: number[][] = [[], [], []];
  for (let col = 0; col < 3; col += 1) {
    b[0][col] = (h[col] - cx * h[col + 6]) / focal;
    b[1][col] = (h[col + 3] - cy * h[col + 6]) / focal;
    b[2][col] = h[col + 6];
  }
  return b;
}

function column(m: number[][], j: number): number[] {
  return [m[0][j], m[1][j], m[2][j]];
}

function norm(v: number[]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function unit(v: number[]): number[] {
  const n = norm(v);
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a: number[], b: number[]): number[] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function orthogonalityResidual(r1: number[], r2: number[]): number {
  const dot = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
  return Math.abs(dot) + Math.abs(1 - norm(cross(r1, r2)));
}

function goldenSection(
  fn: (f: number) => number,
  lo: number,
  hi: number,
): number {
  let a = lo;
  let b = hi;
  let c = b - GOLDEN_RATIO * (b - a);
  let d = a + GOLDEN_RATIO * (b - a);
  let fc = fn(c);
  let fd = fn(d);
  for (let i = 0; i < GOLDEN_ITERATIONS; i += 1) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - GOLDEN_RATIO * (b - a);
      fc = fn(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + GOLDEN_RATIO * (b - a);
      fd = fn(d);
    }
  }
  return (a + b) / 2;
}

export function estimateFocal(
  imageCorners: Point[],
  width: number,
  height: number,
): number {
  const h = findHomography(WORLD_CORNERS, imageCorners);
  const lo = 0.3 * width;
  const hi = 3 * width;
  const residual = (f: number): number => {
    const b = inverseCalibrationHomography(f, width, height, h);
    return orthogonalityResidual(unit(column(b, 0)), unit(column(b, 1)));
  };
  const step = (hi - lo) / SCAN_SAMPLES;
  let bestFocal = lo;
  let bestResidual = residual(lo);
  for (let i = 1; i <= SCAN_SAMPLES; i += 1) {
    const f = lo + i * step;
    const value = residual(f);
    if (value < bestResidual) {
      bestFocal = f;
      bestResidual = value;
    }
  }
  const goldenFocal = goldenSection(residual, lo, hi);
  const goldenResidual = residual(goldenFocal);
  return goldenResidual < bestResidual ? goldenFocal : bestFocal;
}

export function solvePose(
  imageCorners: Point[],
  width: number,
  height: number,
): PlanePose {
  const h = findHomography(WORLD_CORNERS, imageCorners);
  const focal = estimateFocal(imageCorners, width, height);
  const b = inverseCalibrationHomography(focal, width, height, h);
  const scale = norm(column(b, 0));
  const r1 = [b[0][0] / scale, b[1][0] / scale, b[2][0] / scale];
  const r2 = [b[0][1] / scale, b[1][1] / scale, b[2][1] / scale];
  const r3 = cross(r1, r2);
  const metricScale = WHITE_KEY_MM / scale;
  return {
    focal,
    rotation: [
      [r1[0], r2[0], r3[0]],
      [r1[1], r2[1], r3[1]],
      [r1[2], r2[2], r3[2]],
    ],
    translation: [
      b[0][2] * metricScale,
      b[1][2] * metricScale,
      b[2][2] * metricScale,
    ],
    worldWidthMm: WHITE_KEY_COUNT * WHITE_KEY_MM,
    residual: orthogonalityResidual(r1, r2),
  };
}

export function projectPoint(
  pose: PlanePose,
  u: number,
  v: number,
  width: number,
  height: number,
): Point {
  const xmm = u * WHITE_KEY_MM;
  const ymm = v * WHITE_KEY_MM;
  const r = pose.rotation;
  const t = pose.translation;
  const zc = r[2][0] * xmm + r[2][1] * ymm + t[2];
  return {
    x: width / 2 + (pose.focal * (r[0][0] * xmm + r[0][1] * ymm + t[0])) / zc,
    y: height / 2 + (pose.focal * (r[1][0] * xmm + r[1][1] * ymm + t[1])) / zc,
  };
}

export function projectGrid(
  pose: PlanePose,
  width: number,
  height: number,
): GridGeometry {
  const separators: [Point, Point][] = [];
  for (let i = 1; i < WHITE_KEY_COUNT; i += 1) {
    separators.push([
      projectPoint(pose, i, 0, width, height),
      projectPoint(pose, i, DEPTH_UNITS, width, height),
    ]);
  }
  const blackBottomV = BLACK_DEPTH_RATIO * DEPTH_UNITS;
  const blackQuads: Point[][] = [];
  for (let pitch = LOW_PITCH; pitch <= HIGH_PITCH; pitch += 1) {
    if (!isBlack(pitch)) {
      continue;
    }
    const rect = keyRect(pitch);
    const u0 = rect.u0 * WHITE_KEY_COUNT;
    const u1 = rect.u1 * WHITE_KEY_COUNT;
    blackQuads.push([
      projectPoint(pose, u0, 0, width, height),
      projectPoint(pose, u1, 0, width, height),
      projectPoint(pose, u1, blackBottomV, width, height),
      projectPoint(pose, u0, blackBottomV, width, height),
    ]);
  }
  return { separators, blackQuads };
}
