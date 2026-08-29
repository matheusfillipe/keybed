import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { type Calibration, type Corners, createCalibration } from "./calibrate";
import { drawHands } from "./draw";
import { createHandTracker, type HandTracker } from "./hands";
import { createHands3D, type Hands3D } from "./hands3d";
import { findHomography, type Point } from "./homography";
import {
  createScene,
  DEFAULT_ROLL_HEIGHT,
  DEFAULT_ROLL_TILT_DEG,
  type Scene3D,
} from "./scene";
import { createStrip, STRIP_HEIGHT, STRIP_WIDTH, type Strip } from "./strip";

const STRIP_QUAD: Point[] = [
  { x: 0, y: 0 },
  { x: STRIP_WIDTH, y: 0 },
  { x: STRIP_WIDTH, y: STRIP_HEIGHT },
  { x: 0, y: STRIP_HEIGHT },
];

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

function startLoop(
  video: HTMLVideoElement,
  tracker: HandTracker,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  calibration: Calibration,
  strip: Strip,
  scene: Scene3D,
  hands3d: Hands3D,
): void {
  let lastVideoTime = -1;
  let hands: HandLandmarkerResult | null = null;

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
    if (hands) {
      drawHands(ctx, hands, w, h);
    }
    const videoPoints = cornerPoints(calibration.getCorners(), w, h);
    const toStrip = findHomography(videoPoints, STRIP_QUAD);
    const toVideo = findHomography(STRIP_QUAD, videoPoints);
    strip.render(video, toVideo);
    if (hands) {
      hands3d.update(hands, toStrip, w, h);
    }
    scene.render(performance.now());
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
    const strip = createStrip();
    const scene = createScene(strip.canvas);
    const hands3d = createHands3D(scene.root);

    let rollTilt = DEFAULT_ROLL_TILT_DEG;
    let rollHeight = DEFAULT_ROLL_HEIGHT;
    window.addEventListener("keydown", (event) => {
      if (event.key === "1" || event.key === "2") {
        rollTilt = Math.min(
          60,
          Math.max(0, rollTilt + (event.key === "1" ? -5 : 5)),
        );
        scene.setRollTilt(rollTilt);
        console.log(`roll tilt ${rollTilt} deg height ${rollHeight}`);
      }
      if (event.key === "3" || event.key === "4") {
        rollHeight = Math.min(
          50,
          Math.max(10, rollHeight + (event.key === "3" ? -1 : 1)),
        );
        scene.setRollHeight(rollHeight);
        console.log(`roll tilt ${rollTilt} deg height ${rollHeight}`);
      }
    });
    window.addEventListener("resize", () => {
      scene.resize(window.innerWidth, window.innerHeight);
    });

    startLoop(video, tracker, canvas, ctx, calibration, strip, scene, hands3d);
  } catch (err) {
    errorText = errorMessage(err);
    renderError(canvas, errorText);
  }
}

void boot();
