import { describe, expect, test } from "vitest";
import type { Point } from "./homography";
import { checkQuad } from "./quad";

const KEYBED: Point[] = [
  { x: 0.1, y: 0.4 },
  { x: 0.9, y: 0.4 },
  { x: 0.92, y: 0.55 },
  { x: 0.08, y: 0.55 },
];

describe("checkQuad", () => {
  test("accepts a keybed shaped quad", () => {
    expect(checkQuad(KEYBED).usable).toBe(true);
  });

  test("rejects the collapsed quad the detector produced on a miss", () => {
    const collapsed: Point[] = [
      { x: 0.972, y: 0.972 },
      { x: 0.511, y: 0.964 },
      { x: 0.015, y: 0.936 },
      { x: 0.986, y: 0.986 },
    ];
    expect(checkQuad(collapsed).usable).toBe(false);
  });

  test("rejects a quad whose corners fold over", () => {
    const bowtie: Point[] = [
      { x: 0.1, y: 0.4 },
      { x: 0.9, y: 0.4 },
      { x: 0.1, y: 0.55 },
      { x: 0.9, y: 0.55 },
    ];
    expect(checkQuad(bowtie).usable).toBe(false);
  });

  test("rejects a square, which is not a keybed", () => {
    const square: Point[] = [
      { x: 0.2, y: 0.2 },
      { x: 0.7, y: 0.2 },
      { x: 0.7, y: 0.7 },
      { x: 0.2, y: 0.7 },
    ];
    expect(checkQuad(square).usable).toBe(false);
  });

  test("rejects a quad lying across the keys rather than along them", () => {
    const across: Point[] = [
      { x: 0.4, y: 0.1 },
      { x: 0.55, y: 0.1 },
      { x: 0.55, y: 0.9 },
      { x: 0.4, y: 0.9 },
    ];
    expect(checkQuad(across).usable).toBe(false);
  });

  test("explains why it refused", () => {
    expect(checkQuad([KEYBED[0], KEYBED[1], KEYBED[2]]).reason).toBe(
      "needs 4 corners",
    );
  });
});
