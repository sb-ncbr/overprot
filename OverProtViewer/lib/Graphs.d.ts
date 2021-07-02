export declare namespace Graphs {
    type XY = {
        x: number;
        y: number;
    };
    type Size = {
        width: number;
        height: number;
    };
    type Box = {
        left: number;
        right: number;
        top: number;
        bottom: number;
        weight: number;
    };
    export type Dag = {
        levels: number[][];
        edges: [number, number][];
        vertices: number[];
        in_neighbors: Map<number, number[]>;
    };
    export function newDagEmpty(): Dag;
    export function newDagFromPath(vertices: number[]): Dag;
    export function newDagFromPrecedence(levels: number[][], edges: [number, number][]): Dag;
    export function embedDag(dag: Dag, vertexSizes: Map<number, Size>, padding?: XY, leftMargin?: number, rightMargin?: number, topMargin?: number, bottomMargin?: number): [Box, Map<number, XY>];
    export {};
}
