import { Geometry } from './Geometry';
export declare namespace Dag {
    type Dag = {
        nodes: Node[];
        precedence: Edge[];
        beta_connectivity: Edge[];
        levels: number[][];
        precedenceLines: Geometry.Line[];
        activeNodes: number[];
        origPrecedence: Edge[];
        origBetaConnectivity: Edge[];
        error: string | null;
    };
    function newDag(): Dag;
    function dagFromJson(json: string): Dag;
    function newDagWithError(error: string): Dag;
    type Node = {
        label: string;
        occurrence: number;
        avg_length: number;
        type: string;
        sheet_id: number;
        stdev3d: number;
        cdf: number[][];
        active: boolean;
        visual: NodeVisual;
        ladders: number[] | undefined;
    };
    type NodeVisual = {
        rect: Geometry.Rectangle;
        fill: string;
        stroke: string;
    };
    function newNodeVisual(): NodeVisual;
    type Edge = number[];
    function filterDagAndAddLevels(dag: Dag, occurrence_threshold: number): void;
    function getNodeMinMaxLength(node: Dag.Node): [number, number];
}
