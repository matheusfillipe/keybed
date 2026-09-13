import type { Point } from "./homography";
export interface Steady {
    accept(quad: Point[], still?: boolean): Point[];
    reset(): void;
}
export declare function createSteady(): Steady;
