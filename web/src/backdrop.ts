import { CanvasTexture, SRGBColorSpace } from "three";

const SIZE = 512;

export type BackdropKind = "clutter" | "noise" | "gradient";

function context(): CanvasRenderingContext2D {
  const canvas = document.createElement("canvas");
  canvas.width = SIZE;
  canvas.height = SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2d canvas context unavailable");
  }
  return ctx;
}

function clutter(random: () => number): HTMLCanvasElement {
  const ctx = context();
  ctx.fillStyle = `hsl(${random() * 360},${20 + random() * 30}%,${20 + random() * 50}%)`;
  ctx.fillRect(0, 0, SIZE, SIZE);
  for (let i = 0; i < 14; i += 1) {
    ctx.fillStyle = `hsl(${random() * 360},${random() * 40}%,${random() * 80}%)`;
    ctx.fillRect(
      random() * SIZE,
      random() * SIZE,
      random() * SIZE * 0.6,
      random() * SIZE * 0.6,
    );
  }
  return ctx.canvas;
}

function noise(random: () => number): HTMLCanvasElement {
  const ctx = context();
  const image = ctx.createImageData(SIZE, SIZE);
  const scale = 1 + Math.floor(random() * 12);
  for (let y = 0; y < SIZE; y += 1) {
    for (let x = 0; x < SIZE; x += 1) {
      const cell = Math.floor(x / scale) * 73 + Math.floor(y / scale) * 151;
      const value = ((Math.sin(cell) + 1) / 2) * 255;
      const index = (y * SIZE + x) * 4;
      image.data[index] = value;
      image.data[index + 1] = value;
      image.data[index + 2] = value;
      image.data[index + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  return ctx.canvas;
}

function gradient(random: () => number): HTMLCanvasElement {
  const ctx = context();
  const ramp = ctx.createLinearGradient(0, 0, SIZE * random(), SIZE);
  ramp.addColorStop(0, `hsl(${random() * 360},40%,${10 + random() * 40}%)`);
  ramp.addColorStop(1, `hsl(${random() * 360},40%,${10 + random() * 60}%)`);
  ctx.fillStyle = ramp;
  ctx.fillRect(0, 0, SIZE, SIZE);
  return ctx.canvas;
}

export function makeBackdrop(
  kind: BackdropKind,
  random: () => number,
): CanvasTexture {
  const canvas =
    kind === "noise"
      ? noise(random)
      : kind === "gradient"
        ? gradient(random)
        : clutter(random);
  const texture = new CanvasTexture(canvas);
  texture.colorSpace = SRGBColorSpace;
  return texture;
}
