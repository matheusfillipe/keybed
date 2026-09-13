import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { drawHands } from "./draw";
import {
  type Box,
  buildSkinAlpha,
  createSkinSegmenter,
  handBoxes,
  type SkinSegmenter,
} from "./handmask";
import { createHandTracker, type HandTracker } from "./hands";
import { styleButton } from "./hud";
import { viteAssets } from "./viteassets";

const MAX_WIDTH = 640;
const BOX_PADDING = 0.18;
// both models resample to 256 internally, so feeding them a small frame costs nothing in quality
// and saves the per-frame mask copy out of wasm, which scales with the input size
const INFER_WIDTH = 320;
// hands travel far less than a frame between ticks, so the landmark pass can run at half rate
const LANDMARK_EVERY = 2;

interface State {
  handsOnly: boolean;
  skeleton: boolean;
  original: boolean;
}

function panel(state: State, status: HTMLElement): void {
  const bar = document.createElement("div");
  Object.assign(bar.style, {
    position: "fixed",
    top: "12px",
    left: "12px",
    zIndex: "10",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "10px 12px",
    background: "rgba(8,8,10,0.78)",
    border: "1px solid rgba(229,229,229,0.12)",
    borderRadius: "10px",
    color: "#e5e5e5",
    font: "12px/1.5 ui-sans-serif, system-ui, sans-serif",
  });
  const toggle = (
    text: string,
    read: () => boolean,
    write: (on: boolean) => void,
  ): void => {
    const button = document.createElement("button");
    button.textContent = text;
    styleButton(button, read());
    button.addEventListener("click", () => {
      write(!read());
      styleButton(button, read());
    });
    bar.appendChild(button);
  };
  toggle(
    "hands only",
    () => state.handsOnly,
    (on) => {
      state.handsOnly = on;
    },
  );
  toggle(
    "skeleton",
    () => state.skeleton,
    (on) => {
      state.skeleton = on;
    },
  );
  toggle(
    "original",
    () => state.original,
    (on) => {
      state.original = on;
    },
  );
  bar.appendChild(status);
  document.body.appendChild(bar);
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

function loop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  segmenter: SkinSegmenter,
  tracker: HandTracker | null,
  state: State,
  status: HTMLElement,
): void {
  let lastVideoTime = -1;
  let hands: HandLandmarkerResult | null = null;
  let boxes: Box[] = [];
  let smoothed = 0;
  let frames = 0;
  // one scratch canvas, resized only when the camera size changes: assigning width resets a canvas
  const cut = document.createElement("canvas");
  const cutCtx = cut.getContext("2d");
  const small = document.createElement("canvas");
  const smallCtx = small.getContext("2d", { willReadFrequently: false });
  if (!cutCtx || !smallCtx) {
    throw new Error("2d canvas context unavailable");
  }

  const frame = (now: number): void => {
    if (video.videoWidth === 0) {
      requestAnimationFrame(frame);
      return;
    }
    const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
    const width = Math.round(video.videoWidth * scale);
    const height = Math.round(video.videoHeight * scale);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    // a fresh canvas is 300x150, so this is sized on its own dimensions, never on another's
    if (cut.width !== width || cut.height !== height) {
      cut.width = width;
      cut.height = height;
    }

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const started = performance.now();
      frames += 1;
      const inferWidth = Math.min(INFER_WIDTH, video.videoWidth);
      const inferHeight = Math.round(
        (video.videoHeight / video.videoWidth) * inferWidth,
      );
      if (small.width !== inferWidth || small.height !== inferHeight) {
        small.width = inferWidth;
        small.height = inferHeight;
      }
      smallCtx.drawImage(video, 0, 0, inferWidth, inferHeight);
      if (tracker && (state.handsOnly || state.skeleton)) {
        if (frames % LANDMARK_EVERY === 0) {
          hands = tracker.detect(small, now);
          boxes = handBoxes(hands, BOX_PADDING);
        }
      }
      const mask = segmenter.segment(small, now).categoryMask;
      if (mask) {
        const alpha = buildSkinAlpha(
          mask.getAsUint8Array(),
          mask.width,
          mask.height,
          boxes,
          state.handsOnly,
        );
        mask.close();
        cutCtx.globalCompositeOperation = "source-over";
        cutCtx.drawImage(video, 0, 0, width, height);
        if (alpha) {
          // the mask is 256 square; the browser upscale feathers the contour for free
          cutCtx.globalCompositeOperation = "destination-in";
          cutCtx.imageSmoothingEnabled = true;
          cutCtx.drawImage(alpha.canvas, 0, 0, width, height);
        }
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, width, height);
        if (state.original) {
          ctx.globalAlpha = 0.25;
          ctx.drawImage(video, 0, 0, width, height);
          ctx.globalAlpha = 1;
        }
        ctx.drawImage(cut, 0, 0);
        const elapsed = performance.now() - started;
        smoothed = smoothed === 0 ? elapsed : smoothed * 0.9 + elapsed * 0.1;
        status.textContent =
          `${smoothed.toFixed(0)} ms  ${(1000 / Math.max(smoothed, 1)).toFixed(0)} fps  ` +
          `${width}x${height}  frames ${frames}  hands ${boxes.length}  ` +
          `skin ${((alpha?.coverage ?? 0) * 100).toFixed(2)}%`;
      }
    }
    if (hands && state.skeleton) {
      drawHands(ctx, hands, canvas.width, canvas.height);
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

async function boot(): Promise<void> {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.style.display = "none";
  document.body.appendChild(video);

  const canvas = document.createElement("canvas");
  canvas.width = MAX_WIDTH;
  canvas.height = 480;
  Object.assign(canvas.style, {
    position: "fixed",
    inset: "0",
    margin: "auto",
    maxWidth: "100vw",
    maxHeight: "100vh",
  });
  document.body.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("2d canvas context unavailable");
  }

  const state: State = { handsOnly: true, skeleton: false, original: false };
  const status = document.createElement("span");
  status.style.color = "#8a8a8a";
  status.textContent = "loading";
  panel(state, status);

  status.textContent = "loading models";
  const segmenter = await createSkinSegmenter(viteAssets);
  let tracker: HandTracker | null = null;
  try {
    tracker = await createHandTracker(viteAssets);
  } catch (err) {
    status.textContent = `hand model failed: ${err instanceof Error ? err.message : String(err)}`;
  }
  status.textContent = "waiting for camera";
  await startCamera(video);
  status.textContent = "camera ready";
  loop(video, canvas, ctx, segmenter, tracker, state, status);
}

void boot();
