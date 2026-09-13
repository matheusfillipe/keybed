import { type Point } from "./homography";
export declare const ON_KEYBED_MARGIN = 0.1;
export interface Facing {
    quad: Point[];
    margin: number;
    onKeybed: boolean;
}
export declare function facing(gray: Float32Array, size: number, quad: Point[]): Facing;
