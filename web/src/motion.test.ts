import { describe, expect, test } from "vitest";
import { boxAverage, INPUT_SIZE, medianChange } from "./detector";

const SIZE = INPUT_SIZE / 4;

function frame(fill: (x: number, y: number) => number): Float32Array {
  const out = new Float32Array(INPUT_SIZE * INPUT_SIZE);
  for (let y = 0; y < INPUT_SIZE; y += 1) {
    for (let x = 0; x < INPUT_SIZE; x += 1) {
      out[y * INPUT_SIZE + x] = fill(x, y);
    }
  }
  return out;
}

function coarse(gray: Float32Array): Float32Array {
  const out = new Float32Array(SIZE * SIZE);
  boxAverage(gray, out);
  return out;
}

describe("motion gate", () => {
  const scratch = new Float32Array(SIZE * SIZE);
  const scene = frame((x, y) => (((x >> 3) + (y >> 3)) % 2 === 0 ? 0.8 : 0.2));

  test("a hand moving over one patch leaves the median at zero", () => {
    const hand = frame((x, y) =>
      x > 100 && x < 160 && y > 120 && y < 180
        ? 0.5
        : ((x >> 3) + (y >> 3)) % 2 === 0
          ? 0.8
          : 0.2,
    );
    expect(medianChange(coarse(scene), coarse(hand), scratch)).toBe(0);
  });

  test("the camera shifting two pixels moves every pixel", () => {
    const shifted = frame((x, y) =>
      (((x + 2) >> 3) + (y >> 3)) % 2 === 0 ? 0.8 : 0.2,
    );
    expect(
      medianChange(coarse(scene), coarse(shifted), scratch),
    ).toBeGreaterThan(0.006);
  });
});
