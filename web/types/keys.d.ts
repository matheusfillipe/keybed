export declare const LOW_PITCH = 36;
export declare const HIGH_PITCH = 96;
export interface KeyRect {
    u0: number;
    u1: number;
}
export declare const WHITE_COUNT: number;
export declare function isBlack(pitch: number): boolean;
export declare function whiteIndex(pitch: number): number;
export declare function keyRect(pitch: number): KeyRect;
