import type { Point } from "./homography";
export interface QuadCheck {
    usable: boolean;
    reason: string;
}
export declare function checkQuad(quad: readonly Point[]): QuadCheck;
