interface LabRecord {
    t: number;
    ms: number;
    still: boolean;
    motion: number;
    raw: {
        x: number;
        y: number;
    }[] | null;
    drawn: {
        x: number;
        y: number;
    }[] | null;
    note: string;
    points?: {
        x: number;
        y: number;
    }[];
}
declare global {
    interface Window {
        kvtLab?: LabRecord[];
    }
}
export {};
