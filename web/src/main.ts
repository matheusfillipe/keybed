import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { type Calibration, type Corners, createCalibration } from "./calibrate";
import { drawHands } from "./draw";
import { createHandTracker, type HandTracker } from "./hands";
import {
  applyHomography,
  findHomography,
  type Homography,
  type Point,
} from "./homography";
import { pitchAt } from "./keys";
import {
  createStrip,
  drawReadout,
  drawStripBand,
  type FingertipHit,
  STRIP_HEIGHT,
  STRIP_WIDTH,
  type Strip,
} from "./strip";

const FINGERTIP_LANDMARKS = [8, 12];
const STRIP_QUAD: Point[] = [
  { x: 0, y: 0 },
  { x: STRIP_WIDTH, y: 0 },
  { x: STRIP_WIDTH, y: STRIP_HEIGHT },
  { x: 0, y: STRIP_HEIGHT },
];

function cornerPoints(corners: Corners, w: number, h: number): Point[] {
  return corners.map((corner) => ({ x: corner.x * w, y: corner.y * h }));
}

function fingertipHits(
  hands: HandLandmarkerResult,
  toStrip: Homography,
  w: number,
  h: number,
): FingertipHit[] {
  const hits: FingertipHit[] = [];
  for (const [i, landmarks] of hands.landmarks.entries()) {
    const category = hands.handedness[i]?.[0]?.categoryName;
    const label = category === "Left" ? "L" : category === "Right" ? "R" : "?";
    const pitches = new Set<number>();
    for (const index of FINGERTIP_LANDMARKS) {
      const tip = landmarks[index];
      if (!tip) {
        continue;
      }
      const stripPoint = applyHomography(toStrip, tip.x * w, tip.y * h);
      const pitch = pitchAt(stripPoint.x / STRIP_WIDTH);
      if (pitch !== null) {
        pitches.add(pitch);
      }
    }
    for (const pitch of pitches) {
      hits.push({ label, pitch });
    }
  }
  return hits;
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
    const hits = hands ? fingertipHits(hands, toStrip, w, h) : [];
    drawStripBand(ctx, strip.canvas, w, h, hits);
    drawReadout(ctx, hits, h);
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
    startLoop(video, tracker, canvas, ctx, calibration, strip);
  } catch (err) {
    errorText = errorMessage(err);
    renderError(canvas, errorText);
  }
}

void boot();
