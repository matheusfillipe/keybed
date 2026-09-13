import type { Point } from "./homography";
export type Corners = [Point, Point, Point, Point];
export interface Box {
    x: number;
    y: number;
    w: number;
    h: number;
}
export interface Calibration {
    getCorners(): Corners;
    setCorners(corners: readonly Point[]): void;
    draw(ctx: CanvasRenderingContext2D, w: number, h: number): void;
}
export declare function createCalibration(canvas: HTMLCanvasElement, isActive: () => boolean, getBox: () => Box): Calibration;
