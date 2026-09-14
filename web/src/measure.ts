import type { Point } from "./homography";
import { WHITE_KEY_COUNT } from "./pose";
import { checkQuad } from "./quad";
import {
  cameraFocalFraction,
  estimateFocalFraction,
  keybedDepthFromQuad,
  setCameraFocal,
  setKeybedDepth,
} from "./rectfit";

/** Two ends this unequal mean the camera is looking along the keybed rather
 * than down at it. */
const topViewEnds = 1.15;
const depthRange = [3, 9];
const focalRange = [0.45, 2];

/** What corners placed by hand can settle, and it is one or the other: the
 * shape from a view from above, where the aspect is plain to see, and the lens
 * from an oblique view, where the convergence fixes it. */
export type Measurement =
  | { readonly kind: "depth"; readonly units: number }
  | { readonly kind: "focal"; readonly fraction: number }
  | { readonly kind: "refused"; readonly reason: string };

export type Calibration = {
  readonly depthUnits: number | null;
  readonly focalFraction: number | null;
};

export function measureCorners(
  quad: readonly Point[],
  frame: { width: number; height: number },
): Measurement {
  if (!checkQuad(quad).usable) {
    return {
      kind: "refused",
      reason: "put the corners on the keybed, nothing measured",
    };
  }
  const pixels = quad.map((corner) => ({
    x: corner.x * frame.width,
    y: corner.y * frame.height,
  }));
  const end = (a: Point | undefined, b: Point | undefined): number =>
    a === undefined || b === undefined ? 0 : Math.hypot(b.x - a.x, b.y - a.y);
  const ends = [end(pixels[1], pixels[2]), end(pixels[3], pixels[0])];
  const fromAbove = Math.max(...ends) / Math.min(...ends) < topViewEnds;
  if (fromAbove) {
    const units = keybedDepthFromQuad(
      pixels,
      cameraFocalFraction() * frame.width,
      frame.width / 2,
      frame.height / 2,
    );
    return units < (depthRange[0] ?? 0) || units > (depthRange[1] ?? 0)
      ? { kind: "refused", reason: "corners do not make a keybed" }
      : { kind: "depth", units };
  }
  const fraction = estimateFocalFraction(pixels, frame.width, frame.height);
  return fraction < (focalRange[0] ?? 0) || fraction > (focalRange[1] ?? 0)
    ? { kind: "refused", reason: "corners do not make a keybed" }
    : { kind: "focal", fraction };
}

/** How the measurement reads for a person, in key widths per depth. */
export function depthInKeyWidths(units: number): number {
  return WHITE_KEY_COUNT / units;
}

/** The fit reads these globally, so a session applies what it measured once. */
export function applyCalibration(calibration: Calibration): void {
  if (calibration.depthUnits !== null) {
    setKeybedDepth(calibration.depthUnits);
  }
  if (calibration.focalFraction !== null) {
    setCameraFocal(calibration.focalFraction);
  }
}
