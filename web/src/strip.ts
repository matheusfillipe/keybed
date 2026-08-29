import { applyHomography, type Homography } from "./homography";
import { isBlack, KEY_NAMES, keyRect, LOW_PITCH } from "./keys";

export const STRIP_WIDTH = 1280;
export const STRIP_HEIGHT = 160;
const COLUMN_WIDTH_PX = 2;
const BAND_FRACTION = 0.3;

export interface Strip {
  canvas: HTMLCanvasElement;
  render(video: HTMLVideoElement, stripToVideo: Homography): void;
}

export interface FingertipHit {
  label: string;
  pitch: number;
}

export function createStrip(): Strip {
  const canvas = document.createElement("canvas");
  canvas.width = STRIP_WIDTH;
  canvas.height = STRIP_HEIGHT;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2d canvas context unavailable");
  }
  return {
    canvas,
    render: (video, stripToVideo) => {
      ctx.clearRect(0, 0, STRIP_WIDTH, STRIP_HEIGHT);
      for (let x = 0; x < STRIP_WIDTH; x += COLUMN_WIDTH_PX) {
        const top = applyHomography(stripToVideo, x, 0);
        const bottom = applyHomography(stripToVideo, x, STRIP_HEIGHT);
        const sx = Math.min(top.x, bottom.x);
        const sy = Math.min(top.y, bottom.y);
        const sw = Math.max(Math.abs(top.x - bottom.x), 1);
        const sh = Math.max(Math.abs(top.y - bottom.y), 1);
        ctx.drawImage(
          video,
          sx,
          sy,
          sw,
          sh,
          x,
          0,
          COLUMN_WIDTH_PX,
          STRIP_HEIGHT,
        );
      }
    },
  };
}

function bandTop(height: number): number {
  return Math.round(height * (1 - BAND_FRACTION));
}

export function drawStripBand(
  ctx: CanvasRenderingContext2D,
  strip: HTMLCanvasElement,
  width: number,
  height: number,
  hits: FingertipHit[],
): void {
  const top = bandTop(height);
  const bandHeight = height - top;
  ctx.drawImage(strip, 0, top, width, bandHeight);
  ctx.globalCompositeOperation = "lighter";
  for (const hit of hits) {
    const rect = keyRect(hit.pitch);
    ctx.fillStyle = isBlack(hit.pitch)
      ? "rgba(232,121,249,0.4)"
      : "rgba(34,211,238,0.4)";
    ctx.fillRect(rect.u0 * width, top, (rect.u1 - rect.u0) * width, bandHeight);
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "rgba(229,229,229,0.2)";
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, top + 0.5, width - 1, bandHeight - 1);
}

const READOUT_COLORS: Record<string, string> = {
  L: "#38bdf8",
  R: "#f472b6",
};

export function drawReadout(
  ctx: CanvasRenderingContext2D,
  hits: FingertipHit[],
  height: number,
): void {
  const byHand = new Map<string, number[]>();
  for (const hit of hits) {
    const pitches = byHand.get(hit.label);
    if (pitches) {
      if (!pitches.includes(hit.pitch)) {
        pitches.push(hit.pitch);
      }
    } else {
      byHand.set(hit.label, [hit.pitch]);
    }
  }
  ctx.font = "14px sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  let line = 0;
  for (const [label, pitches] of byHand) {
    const names = pitches
      .map((pitch) => KEY_NAMES[pitch - LOW_PITCH])
      .join("  ");
    ctx.fillStyle = READOUT_COLORS[label] ?? "#e5e5e5";
    ctx.fillText(`${label}: ${names}`, 12, bandTop(height) - 10 - line * 20);
    line += 1;
  }
}
