import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { type Calibration, type Corners, createCalibration } from "./calibrate";
import { drawHands } from "./draw";
import { createHandTracker, type HandTracker } from "./hands";
import type { Point } from "./homography";
import { createLab } from "./lab";
import { type GridGeometry, projectGrid, solvePose } from "./pose";

function cornerPoints(corners: Corners, w: number, h: number): Point[] {
  return corners.map((corner) => ({ x: corner.x * w, y: corner.y * h }));
}

function createVideo(): HTMLVideoElement {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.style.display = "none";
  document.body.appendChild(video);
  return video;
}

function createCanvas(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.style.position = "fixed";
  canvas.style.inset = "0";
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
  document.body.appendChild(canvas);
  return canvas;
}

function renderError(canvas: HTMLCanvasElement, message: string): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#e5e5e5";
  ctx.font = "16px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function errorMessage(err: unknown): string {
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    return "camera access denied";
  }
  return err instanceof Error ? err.message : String(err);
}

async function startCamera(video: HTMLVideoElement): Promise<void> {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: false,
  });
  video.srcObject = stream;
  await new Promise<void>((resolve) => {
    video.addEventListener("loadeddata", () => resolve(), { once: true });
  });
  await video.play();
}

function drawGrid(ctx: CanvasRenderingContext2D, grid: GridGeometry): void {
  ctx.fillStyle = "rgba(10,10,10,0.55)";
  ctx.strokeStyle = "rgba(229,229,229,0.15)";
  ctx.lineWidth = 1;
  for (const quad of grid.blackQuads) {
    ctx.beginPath();
    for (const [i, p] of quad.entries()) {
      if (i === 0) {
        ctx.moveTo(p.x, p.y);
      } else {
        ctx.lineTo(p.x, p.y);
      }
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(229,229,229,0.5)";
  ctx.beginPath();
  for (const [a, b] of grid.separators) {
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
  }
  ctx.stroke();
}

function startLoop(
  video: HTMLVideoElement,
  tracker: HandTracker,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  calibration: Calibration,
): void {
  let lastVideoTime = -1;
  let hands: HandLandmarkerResult | null = null;
  let gridVisible = true;

  window.addEventListener("keydown", (event) => {
    if (event.key === "g") {
      gridVisible = !gridVisible;
    }
  });

  const frame = (): void => {
    const w = canvas.width;
    const h = canvas.height;
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hands = tracker.detect(video, performance.now());
    }
    ctx.drawImage(video, 0, 0, w, h);
    ctx.fillStyle = "rgba(5,5,5,0.35)";
    ctx.fillRect(0, 0, w, h);
    if (gridVisible) {
      const pose = solvePose(
        cornerPoints(calibration.getCorners(), w, h),
        w,
        h,
      );
      drawGrid(ctx, projectGrid(pose, w, h));
    }
    if (hands) {
      drawHands(ctx, hands, w, h);
    }
    calibration.draw(ctx, w, h);
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

async function boot(): Promise<void> {
  const video = createVideo();
  const canvas = createCanvas();
  let errorText: string | null = null;

  const resize = (): void => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    if (errorText) {
      renderError(canvas, errorText);
    }
  };
  resize();
  window.addEventListener("resize", resize);

  try {
    const [tracker] = await Promise.all([
      createHandTracker(),
      startCamera(video),
    ]);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2d canvas context unavailable");
    }
    const calibration = createCalibration(canvas);
    startLoop(video, tracker, canvas, ctx, calibration);
    const stream = video.srcObject;
    if (stream instanceof MediaStream) {
      createLab({ video, stream, getCorners: calibration.getCorners });
    }
  } catch (err) {
    errorText = errorMessage(err);
    renderError(canvas, errorText);
  }
}

void boot();
