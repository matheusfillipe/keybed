import { beforeEach, describe, expect, test } from "vitest";
import { type Box, createCalibration } from "./calibrate";

// the suite runs headless, and calibration persists corners
const store = new Map<string, string>();
globalThis.localStorage = {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => {
    store.set(k, v);
  },
  removeItem: (k: string) => {
    store.delete(k);
  },
  clear: () => {
    store.clear();
  },
  key: () => null,
  length: 0,
} as Storage;

// the video is letterboxed inside the canvas, so the bars offset every handle
const CANVAS = { width: 1000, height: 600 };
const BOX: Box = { x: 100, y: 0, w: 800, h: 600 };

function harness(): {
  canvas: HTMLCanvasElement;
  calibration: ReturnType<typeof createCalibration>;
} {
  const listeners = new Map<string, (event: PointerEvent) => void>();
  const canvas = {
    width: CANVAS.width,
    height: CANVAS.height,
    addEventListener: (name: string, fn: (event: PointerEvent) => void) => {
      listeners.set(name, fn);
    },
    setPointerCapture: () => {},
  } as unknown as HTMLCanvasElement;
  const calibration = createCalibration(
    canvas,
    () => true,
    () => BOX,
  );
  Object.assign(canvas, {
    fire: (name: string, x: number, y: number) =>
      listeners.get(name)?.({
        clientX: x,
        clientY: y,
        pointerId: 1,
      } as PointerEvent),
  });
  return { canvas, calibration };
}

function fire(canvas: HTMLCanvasElement, name: string, x: number, y: number) {
  (
    canvas as unknown as { fire: (n: string, x: number, y: number) => void }
  ).fire(name, x, y);
}

describe("corner handles", () => {
  beforeEach(() => localStorage.clear());

  test("a handle is grabbed where it is drawn, not where the canvas would put it", () => {
    const { canvas, calibration } = harness();
    const corner = calibration.getCorners()[0];
    fire(
      canvas,
      "pointerdown",
      BOX.x + corner.x * BOX.w,
      BOX.y + corner.y * BOX.h,
    );
    fire(canvas, "pointermove", BOX.x + 0.5 * BOX.w, BOX.y + 0.5 * BOX.h);
    expect(calibration.getCorners()[0].x).toBeCloseTo(0.5, 5);
    expect(calibration.getCorners()[0].y).toBeCloseTo(0.5, 5);
  });

  test("a press inside the letterbox bar grabs nothing", () => {
    const { canvas, calibration } = harness();
    const before = { ...calibration.getCorners()[0] };
    fire(canvas, "pointerdown", 10, 10);
    fire(canvas, "pointermove", BOX.x + 0.5 * BOX.w, BOX.y + 0.5 * BOX.h);
    expect(calibration.getCorners()[0]).toEqual(before);
  });

  test("a drag is reported in frame fractions the label can use", () => {
    const { canvas, calibration } = harness();
    const corner = calibration.getCorners()[2];
    fire(
      canvas,
      "pointerdown",
      BOX.x + corner.x * BOX.w,
      BOX.y + corner.y * BOX.h,
    );
    fire(canvas, "pointermove", BOX.x + BOX.w, BOX.y + BOX.h);
    expect(calibration.getCorners()[2]).toEqual({ x: 1, y: 1 });
  });
});
