import { type Point } from "./homography";
export declare const WHITE_KEY_MM = 23.5;
export declare const KEYBED_DEPTH_MM = 118;
export declare const WHITE_KEY_COUNT: number;
export declare const DEPTH_UNITS: number;
export interface PlanePose {
    focal: number;
    rotation: number[][];
    translation: number[];
    worldWidthMm: number;
    residual: number;
}
export declare function canonicalQuad(quad: Point[]): Point[];
export declare function estimateFocal(imageCorners: Point[], width: number, height: number): number;
export declare function solvePose(imageCorners: Point[], width: number, height: number): PlanePose;
export declare function projectPoint(pose: PlanePose, u: number, v: number, width: number, height: number): Point;
