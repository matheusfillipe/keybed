import { PerspectiveCamera, Vector3 } from "three";
import { describe, expect, test } from "vitest";
import {
  DEPTH_UNITS,
  type PlanePose,
  projectPoint,
  WHITE_KEY_COUNT,
  WHITE_KEY_MM,
} from "./pose";
import { applyPose } from "./scene";

const WIDTH = 1280;
const HEIGHT = 720;

type Mat3 = number[][];

function rotationX(a: number): Mat3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rotationY(a: number): Mat3 {
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

function matMul(a: Mat3, b: Mat3): Mat3 {
  const out: Mat3 = [];
  for (let i = 0; i < 3; i += 1) {
    const row: number[] = [];
    for (let j = 0; j < 3; j += 1) {
      row.push(a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]);
    }
    out.push(row);
  }
  return out;
}

function applyCv(
  pose: PlanePose,
  p: [number, number, number],
): { x: number; y: number } {
  const r = pose.rotation;
  const t = pose.translation;
  const cx = r[0][0] * p[0] + r[0][1] * p[1] + r[0][2] * p[2] + t[0];
  const cy = r[1][0] * p[0] + r[1][1] * p[1] + r[1][2] * p[2] + t[1];
  const cz = r[2][0] * p[0] + r[2][1] * p[1] + r[2][2] * p[2] + t[2];
  return {
    x: WIDTH / 2 + (pose.focal * cx) / cz,
    y: HEIGHT / 2 + (pose.focal * cy) / cz,
  };
}

function projectThroughCamera(
  camera: PerspectiveCamera,
  p: [number, number, number],
): { x: number; y: number } {
  const v = new Vector3(p[0], p[1], p[2]).project(camera);
  return { x: ((v.x + 1) / 2) * WIDTH, y: ((1 - v.y) / 2) * HEIGHT };
}

describe("scene camera", () => {
  test("projects world points onto the solved 2D pose projection", () => {
    const rotation = matMul(rotationX(-0.32), rotationY(0.18));
    const c = applyCv(
      {
        focal: 800,
        rotation,
        translation: [0, 0, 0],
        worldWidthMm: 0,
        residual: 0,
      },
      [
        (WHITE_KEY_COUNT * WHITE_KEY_MM) / 2,
        (DEPTH_UNITS * WHITE_KEY_MM) / 2,
        0,
      ],
    );
    const pose: PlanePose = {
      focal: 800,
      rotation,
      translation: [-c.x, -c.y, 800],
      worldWidthMm: WHITE_KEY_COUNT * WHITE_KEY_MM,
      residual: 0,
    };
    const camera = new PerspectiveCamera();
    applyPose(camera, pose, WIDTH, HEIGHT);
    const points: [number, number, number][] = [
      [0, 0, 0],
      [WHITE_KEY_COUNT * WHITE_KEY_MM, 0, 0],
      [WHITE_KEY_COUNT * WHITE_KEY_MM, DEPTH_UNITS * WHITE_KEY_MM, 0],
      [0, DEPTH_UNITS * WHITE_KEY_MM, 0],
      [
        (WHITE_KEY_COUNT / 2) * WHITE_KEY_MM,
        (DEPTH_UNITS / 2) * WHITE_KEY_MM,
        0,
      ],
      [300, 75, 240],
      [900, 40, 600],
    ];
    for (const p of points) {
      const expected =
        p[2] === 0
          ? projectPoint(
              pose,
              p[0] / WHITE_KEY_MM,
              p[1] / WHITE_KEY_MM,
              WIDTH,
              HEIGHT,
            )
          : applyCv(pose, p);
      const got = projectThroughCamera(camera, p);
      expect(Math.abs(got.x - expected.x)).toBeLessThan(1e-4);
      expect(Math.abs(got.y - expected.y)).toBeLessThan(1e-4);
    }
  });
});
