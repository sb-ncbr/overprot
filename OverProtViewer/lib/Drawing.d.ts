import { Types } from './Types';
import { Dag } from './Dag';
export declare namespace Drawing {
    function zoomOut(viewer: Types.Viewer, ratio?: number, mouseXY?: number[]): void;
    function zoomIn(viewer: Types.Viewer, ratio?: number, mouseXY?: number[]): void;
    function zoomAll(viewer: Types.Viewer): void;
    function zoomSet(viewer: Types.Viewer, newZoomout: number, centerXY?: [number, number]): void;
    function shift(viewer: Types.Viewer, rightRelative: number, downRelative: number): void;
    function shiftByMouse(viewer: Types.Viewer, rightPixels: number, downPixels: number): void;
    function setTooltips(viewer: Types.Viewer, selection: Types.D3Selection, htmlContents: (string | null)[] | null, pinnable?: boolean, delay?: boolean): void;
    function addPointBehavior(selection: Types.D3Selection, pointedElementSelector?: (pointed: HTMLElement) => Types.D3Selection): void;
    function addPickBehavior(viewer: Types.Viewer, selection: Types.D3Selection, pickedElementSelector?: (clicked: HTMLElement) => Types.D3Selection): void;
    function addMouseHoldBehavior(selection: Types.D3Selection, onDown: () => any, onHold: () => any, onUp: () => any): void;
    function recolor(viewer: Types.Viewer, transition?: boolean): void;
    function redraw(viewer: Types.Viewer, transition?: boolean): void;
    function nodeBigEnoughForLabel(viewer: Types.Viewer, node: Dag.Node): boolean;
    function showBetaConnectivity(viewer: Types.Viewer, on: boolean, transition?: boolean): void;
    function fadeOutRemove(selection: Types.D3Selection, delay?: number): Types.D3Transition;
    function fadeIn(selection: Types.D3Selection, delay?: number): Types.D3Transition;
}
