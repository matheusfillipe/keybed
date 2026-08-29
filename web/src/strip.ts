import { applyHomography, type Homography } from "./homography";

export const STRIP_WIDTH = 1280;
export const STRIP_HEIGHT = 160;
const COLUMN_WIDTH_PX = 2;

export interface Strip {
  canvas: HTMLCanvasElement;
  render(video: HTMLVideoElement, stripToVideo: Homography): void;
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
