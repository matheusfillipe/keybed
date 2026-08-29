import type { Point } from "./homography";

const STORAGE_KEY = "kvt-corners";
const HIT_RADIUS_PX = 24;
const HANDLE_RADIUS_PX = 9;
const HINT_TEXT = "drag corners onto the keyboard, press c to hide";

export type Corners = [Point, Point, Point, Point];

export interface Calibration {
  getCorners(): Corners;
  draw(ctx: CanvasRenderingContext2D, w: number, h: number): void;
}

function defaultCorners(): Corners {
  return [
    { x: 0.25, y: 0.3 },
    { x: 0.75, y: 0.3 },
    { x: 0.75, y: 0.7 },
    { x: 0.25, y: 0.7 },
  ];
}

function isPoint(value: unknown): value is Point {
  return (
    typeof value === "object" &&
    value !== null &&
    "x" in value &&
    "y" in value &&
    typeof value.x === "number" &&
    Number.isFinite(value.x) &&
    typeof value.y === "number" &&
    Number.isFinite(value.y)
  );
}

function loadCorners(): Corners | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
  if (!Array.isArray(parsed) || parsed.length !== 4) {
    return null;
  }
  const points: Point[] = [];
  for (const item of parsed) {
    if (!isPoint(item)) {
      return null;
    }
    points.push(item);
  }
  const corners: Corners = [points[0], points[1], points[2], points[3]];
  return corners;
}

function saveCorners(corners: Corners): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(corners));
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function createCalibration(canvas: HTMLCanvasElement): Calibration {
  const saved = loadCorners();
  const corners: Corners = saved ?? defaultCorners();
  let calibrating = saved === null;
  let dragging: number | null = null;

  const hitCorner = (px: number, py: number): number | null => {
    let best: number | null = null;
    let bestDist = HIT_RADIUS_PX;
    for (const [i, corner] of corners.entries()) {
      const dx = px - corner.x * canvas.width;
      const dy = py - corner.y * canvas.height;
      const dist = Math.hypot(dx, dy);
      if (dist <= bestDist) {
        best = i;
        bestDist = dist;
      }
    }
    return best;
  };

  const release = (): void => {
    if (dragging === null) {
      return;
    }
    dragging = null;
    saveCorners(corners);
  };

  canvas.addEventListener("pointerdown", (event) => {
    if (!calibrating || dragging !== null) {
      return;
    }
    const index = hitCorner(event.clientX, event.clientY);
    if (index === null) {
      return;
    }
    dragging = index;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (dragging === null) {
      return;
    }
    corners[dragging] = {
      x: clamp01(event.clientX / canvas.width),
      y: clamp01(event.clientY / canvas.height),
    };
  });
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  window.addEventListener("keydown", (event) => {
    if (event.key === "c") {
      calibrating = !calibrating;
    }
  });

  return {
    getCorners: () => corners,
    draw: (ctx, w, h) => {
      if (!calibrating) {
        return;
      }
      ctx.strokeStyle = "rgba(56,189,248,0.9)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (const [i, corner] of corners.entries()) {
        const px = corner.x * w;
        const py = corner.y * h;
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.closePath();
      ctx.stroke();
      for (const [i, corner] of corners.entries()) {
        const px = corner.x * w;
        const py = corner.y * h;
        ctx.beginPath();
        ctx.arc(px, py, HANDLE_RADIUS_PX, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(229,229,229,0.95)";
        ctx.fill();
        ctx.fillStyle = "#050505";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), px, py);
      }
      if (saved === null) {
        ctx.fillStyle = "#e5e5e5";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(HINT_TEXT, w / 2, 12);
      }
    },
  };
}
