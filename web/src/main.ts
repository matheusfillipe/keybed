import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { type Calibration, type Corners, createCalibration } from "./calibrate";
import { createDetector, type Detector, INPUT_SIZE } from "./detector";
import { drawHands, drawModelInput, drawQuad } from "./draw";
import { createHandTracker, type HandTracker } from "./hands";
import type { Point } from "./homography";
import { createHud, type Hud } from "./hud";
import { createLab } from "./lab";
import { facing } from "./orient";
import { canonicalQuad, solvePose, WHITE_KEY_COUNT } from "./pose";
import { checkQuad } from "./quad";
import {
  cameraFocalFraction,
  estimateFocalFraction,
  keybedDepthFromQuad,
  setCameraFocal,
  setKeybedDepth,
} from "./rectfit";
import { createSteady } from "./steady";
import { viteAssets } from "./viteassets";

const DETECT_INTERVAL_MS = 250;
const MANUAL_COLOR = "rgba(56,189,248,0.9)";
const AUTO_COLOR = "#4ade80";
const WEAK_COLOR = "#f87171";
const MODEL_VIEW_PX = 216;
// a couple of rejected frames is noise; a run of them means the keybed really is gone
const MISSES_BEFORE_CLEAR = 4;
// measured over every labelled frame: a quad that is really a perspective view of the keybed
// solves with residual under 0.32; a broken one scores 2.0 or more; a well shaped quad in the
// wrong place lands between, and this catches the worse half of those too
const MAX_POSE_RESIDUAL = 1.0;
// the mask inside a real keybed's box averages near 1; a box drawn over a guess averages
// far less, and a guess is not shown at all
const MIN_CONFIDENCE = 0.6;
const DEPTH_KEY = "kvt.keybedDepthUnits.v2";
const FOCAL_KEY = "kvt.cameraFocalFraction.v2";
// ends within this ratio of each other are a view from above; further apart is oblique
const TOP_VIEW_ENDS = 1.15;
// a keybed is 3 to 9 key widths deep and a camera lens 0.45 to 2 frame widths of focal;
// corners that measure outside that were not on the keybed and measure nothing
const DEPTH_RANGE = [3, 9];
const FOCAL_RANGE = [0.45, 2];

// when a recording plays in place of the camera, every detection is kept on the window so
// a lab session can read the pipeline's behaviour over time
interface LabRecord {
  t: number;
  ms: number;
  still: boolean;
  motion: number;
  raw: { x: number; y: number }[] | null;
  drawn: { x: number; y: number }[] | null;
  note: string;
  // the snapped boundary points the rectangle fit was given, in frame pixels
  points?: { x: number; y: number }[];
}
declare global {
  interface Window {
    kvtLab?: LabRecord[];
  }
}
function labLog(record: LabRecord): void {
  window.kvtLab?.push(record);
}

interface Lock {
  quad: Point[];
  inputQuad: Point[];
  latencyMs: number;
  onKeybed: boolean;
  margin: number;
  gray: Float32Array;
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
  ctx.font = "16px ui-sans-serif, system-ui, sans-serif";
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
  // ?clip=<recording>.webm plays a saved recording in place of the camera, so the whole
  // pipeline can be watched in a browser with no camera at all
  const clip = new URLSearchParams(location.search).get("clip");
  if (clip) {
    video.src = `/lab/clip/${clip}`;
    video.loop = true;
    video.muted = true;
    window.kvtLab = [];
  } else {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
  }
  await new Promise<void>((resolve) => {
    video.addEventListener("loadeddata", () => resolve(), { once: true });
  });
  await video.play();
}

// the video fills the canvas without stretching, so the overlay sits on the pixels the model saw
function videoBox(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): { x: number; y: number; w: number; h: number } {
  const scale = Math.min(
    canvas.width / video.videoWidth,
    canvas.height / video.videoHeight,
  );
  const w = video.videoWidth * scale;
  const h = video.videoHeight * scale;
  return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
}

function startLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  calibration: Calibration,
  hud: Hud,
  detector: Detector | null,
  tracker: HandTracker | null,
): () => Corners {
  let lock: Lock | null = null;
  let misses = 0;
  const steady = createSteady();
  let hands: HandLandmarkerResult | null = null;
  let lastVideoTime = -1;
  let inFlight = false;
  let lastDetectAt = -Infinity;

  const detect = (now: number): void => {
    if (!detector || inFlight || video.videoWidth === 0) {
      return;
    }
    inFlight = true;
    lastDetectAt = now;
    detector
      .detect(video)
      .then((detection) => {
        if (!detection.quad) {
          labLog({
            t: performance.now(),
            ms: detection.latencyMs,
            still: detection.still,
            motion: detection.motion,
            raw: null,
            drawn: null,
            note: "no keybed",
          });
          lock = null;
          steady.reset();
          hud.status("detect", `${detection.latencyMs.toFixed(1)} ms`);
          hud.status(
            "keybed",
            `no keybed (mask ${(detection.coverage * 100).toFixed(1)}%)`,
          );
          return;
        }
        // the fit's corner order says nothing about which long edge is the back; the
        // picture does: the black keys are darker than the white key fronts, so a quad
        // whose "back" band is the lighter one is turned around
        const asDetected = facing(
          detection.gray,
          INPUT_SIZE,
          detection.inputQuad ?? detection.quad,
        );
        const reversed = asDetected.margin < 0;
        const turned = (q: Point[]): Point[] =>
          reversed ? [q[2], q[3], q[0], q[1]] : q;
        const facts = reversed
          ? facing(
              detection.gray,
              INPUT_SIZE,
              turned(detection.inputQuad ?? detection.quad),
            )
          : asDetected;
        const framed = canonicalQuad(turned(canonicalQuad(detection.quad)));
        // a quad that is not keybed shaped is a miss. Leaving the last good one on screen
        // reads as a frozen detection, so a run of misses clears it instead.
        const shape = checkQuad(framed);
        const residual = shape.usable
          ? solvePose(
              framed.map((p) => ({
                x: p.x * video.videoWidth,
                y: p.y * video.videoHeight,
              })),
              video.videoWidth,
              video.videoHeight,
            ).residual
          : 0;
        // a box the mask does not fill, or without the black-key stripe, is a guess at a
        // keybed and is not shown: nothing on screen beats the most likely wrong thing
        const unsure = detection.confidence < MIN_CONFIDENCE || !facts.onKeybed;
        if (!shape.usable || residual > MAX_POSE_RESIDUAL || unsure) {
          labLog({
            t: performance.now(),
            ms: detection.latencyMs,
            still: detection.still,
            motion: detection.motion,
            raw: framed,
            drawn: null,
            note: shape.usable
              ? `residual ${residual.toFixed(2)}`
              : shape.reason,
          });
          misses += 1;
          if (misses >= MISSES_BEFORE_CLEAR) {
            lock = null;
            steady.reset();
          }
          hud.status("detect", `${detection.latencyMs.toFixed(1)} ms`);
          hud.status(
            "keybed",
            `${
              !shape.usable
                ? shape.reason
                : residual > MAX_POSE_RESIDUAL
                  ? `not a rectangle in 3d (${residual.toFixed(1)})`
                  : detection.confidence < MIN_CONFIDENCE
                    ? `mask ${(detection.confidence * 100).toFixed(0)}% inside, not a keybed`
                    : "no black-key stripe, not a keybed"
            } (${misses} missed)`,
          );
          return;
        }
        misses = 0;
        lock = {
          quad: steady.accept(framed, detection.still),
          inputQuad: facts.quad,
          latencyMs: detection.latencyMs,
          onKeybed: facts.onKeybed,
          margin: facts.margin,
          gray: detection.gray,
        };
        labLog({
          t: performance.now(),
          ms: detection.latencyMs,
          still: detection.still,
          motion: detection.motion,
          raw: framed,
          drawn: lock.quad,
          note: "",
          points: window.kvtPoints,
        });
        hud.status(
          "detect",
          `${detection.latencyMs.toFixed(1)} ms  ${detection.still ? "still" : "moving"} ${
            Number.isFinite(detection.motion)
              ? detection.motion.toFixed(4)
              : "-"
          }`,
        );
        hud.status(
          "keybed",
          `mask ${(detection.coverage * 100).toFixed(1)}%  ` +
            `${facts.onKeybed ? "keys found" : "no key pattern"} (${facts.margin.toFixed(2)})`,
        );
      })
      .catch((err: unknown) => {
        hud.status("model", errorMessage(err));
      })
      .finally(() => {
        inFlight = false;
      });
  };

  hud.onRedetect(() => detect(performance.now()));
  hud.onAdopt(() => {
    if (lock) {
      calibration.setCorners(lock.quad);
    }
  });
  // the corners are often placed perfectly but a half turn out, which reads as front and back
  // swapped; rolling by two relabels the same rectangle rather than making it be dragged again
  hud.onFlip(() => {
    const c = calibration.getCorners();
    calibration.setCorners([c[2], c[3], c[0], c[1]]);
  });
  // the keybed's shape and the camera's lens, each measured once from the dragged corners
  // and kept in this browser: the shape from a view from above, where the aspect is plain
  // to see, the lens from an oblique view, where the convergence fixes it. Which one a
  // press measures is decided by how unequal the two ends are.
  try {
    const depth = localStorage.getItem(DEPTH_KEY);
    if (depth) {
      setKeybedDepth(Number(depth));
    }
    const focal = localStorage.getItem(FOCAL_KEY);
    if (focal) {
      setCameraFocal(Number(focal));
    }
  } catch {
    // storage can be blocked; the defaults stand
  }
  hud.onMeasure(() => {
    const px = orientedManual().map((p) => ({
      x: p.x * video.videoWidth,
      y: p.y * video.videoHeight,
    }));
    if (!checkQuad(orientedManual()).usable) {
      hud.status("keybed", "put the corners on the keybed, nothing measured");
      return;
    }
    const end = (a: Point, b: Point): number =>
      Math.hypot(b.x - a.x, b.y - a.y);
    const ends = [end(px[1], px[2]), end(px[3], px[0])];
    const fromAbove = Math.max(...ends) / Math.min(...ends) < TOP_VIEW_ENDS;
    if (fromAbove) {
      const units = keybedDepthFromQuad(
        px,
        cameraFocalFraction() * video.videoWidth,
        video.videoWidth / 2,
        video.videoHeight / 2,
      );
      if (units < DEPTH_RANGE[0] || units > DEPTH_RANGE[1]) {
        hud.status("keybed", "corners do not make a keybed, nothing measured");
        return;
      }
      setKeybedDepth(units);
      try {
        localStorage.setItem(DEPTH_KEY, String(units));
      } catch {
        // storage can be blocked; the measurement holds for this session
      }
      hud.status(
        "keybed",
        `shape measured: ${(WHITE_KEY_COUNT / units).toFixed(2)} key widths per depth`,
      );
      return;
    }
    const fraction = estimateFocalFraction(
      px,
      video.videoWidth,
      video.videoHeight,
    );
    if (fraction < FOCAL_RANGE[0] || fraction > FOCAL_RANGE[1]) {
      hud.status("keybed", "corners do not make a keybed, nothing measured");
      return;
    }
    setCameraFocal(fraction);
    try {
      localStorage.setItem(FOCAL_KEY, String(fraction));
    } catch {
      // storage can be blocked; the measurement holds for this session
    }
    hud.status(
      "keybed",
      `lens measured: focal ${fraction.toFixed(2)} of the frame width`,
    );
  });

  // the dragged corners mean what was dragged: handle 1 to 2 runs along the black keys,
  // and the flip button is how the back is swapped. Deciding the back from the picture
  // every frame made the handles jump between the two orders on their own.
  const orientedManual = (): Corners =>
    canonicalQuad(calibration.getCorners()) as Corners;

  const frame = (now: number): void => {
    if (hud.state.live && now - lastDetectAt > DETECT_INTERVAL_MS) {
      detect(now);
    }
    if (tracker && hud.state.hands && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hands = tracker.detect(video, now);
    }
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const box = videoBox(video, canvas);
    ctx.drawImage(video, box.x, box.y, box.w, box.h);

    ctx.save();
    ctx.translate(box.x, box.y);
    if (lock) {
      drawQuad(
        ctx,
        lock.quad,
        box.w,
        box.h,
        lock.onKeybed ? AUTO_COLOR : WEAK_COLOR,
        lock.onKeybed ? "keybed" : "no key pattern",
      );
    }
    if (hud.state.corners) {
      drawQuad(ctx, orientedManual(), box.w, box.h, MANUAL_COLOR, "manual");
      calibration.draw(ctx, box.w, box.h);
    }
    if (hands && hud.state.hands) {
      drawHands(ctx, hands, box.w, box.h);
    }
    ctx.restore();

    if (lock && hud.state.input) {
      drawModelInput(ctx, lock.gray, INPUT_SIZE, lock.inputQuad, MODEL_VIEW_PX);
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return orientedManual;
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
    const hud = createHud();
    hud.status("model", "loading");
    const loading = createDetector(viteAssets).then(
      (detector) => {
        hud.status("model", "ready");
        return detector;
      },
      (err: unknown) => {
        hud.status("model", `unavailable (${errorMessage(err)})`);
        return null;
      },
    );
    await startCamera(video);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2d canvas context unavailable");
    }
    const calibration = createCalibration(
      canvas,
      () => hud.state.corners,
      () => videoBox(video, canvas),
    );
    const tracker = await createHandTracker(viteAssets).catch(() => null);
    const labelCorners = startLoop(
      video,
      canvas,
      ctx,
      calibration,
      hud,
      await loading,
      tracker,
    );
    const stream = video.srcObject;
    if (stream instanceof MediaStream) {
      createLab({
        video,
        stream,
        getCorners: labelCorners,
        mount: hud.capture,
      });
    }
  } catch (err) {
    errorText = errorMessage(err);
    renderError(canvas, errorText);
  }
}

void boot();
