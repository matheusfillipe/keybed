import type { Point } from "./homography";
/** What corners placed by hand can settle, and it is one or the other: the
 * shape from a view from above, where the aspect is plain to see, and the lens
 * from an oblique view, where the convergence fixes it. */
export type Measurement = {
    readonly kind: "depth";
    readonly units: number;
} | {
    readonly kind: "focal";
    readonly fraction: number;
} | {
    readonly kind: "refused";
    readonly reason: string;
};
export type Calibration = {
    readonly depthUnits: number | null;
    readonly focalFraction: number | null;
};
export declare function measureCorners(quad: readonly Point[], frame: {
    width: number;
    height: number;
}): Measurement;
/** How the measurement reads for a person, in key widths per depth. */
export declare function depthInKeyWidths(units: number): number;
/** The fit reads these globally, so a session applies what it measured once. */
export declare function applyCalibration(calibration: Calibration): void;
