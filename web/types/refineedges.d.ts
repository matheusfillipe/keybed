import type { Point } from "./homography";
export declare const SEARCH_PX = 10;
export declare const SEARCH_STEP = 0.5;
export declare const MIN_RESPONSE = 6;
export declare function blur(gray: Float32Array, width: number, height: number): Float32Array;
export declare function bilinear(gray: Float32Array, width: number, height: number, x: number, y: number): number;
export declare function refineQuad(gray: Float32Array, width: number, height: number, quad: Point[], searchPx?: number): Point[];
