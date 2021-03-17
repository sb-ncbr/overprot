import * as d3 from 'd3';
export declare namespace Geometry {
    type Rectangle = {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    function newRectangle(): Rectangle;
    function rectangleFromCanvas(canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>): Rectangle;
    function rectangleCenter(rect: Rectangle): [number, number];
    type Line = {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
    };
    type Polyline = number[][];
    type ZoomInfo = {
        zoomout: number;
        xZoomout: number;
        yZoomout: number;
        minZoomout: number;
        maxZoomout: number;
        minXZoomout: number;
        maxXZoomout: number;
        minYZoomout: number;
        maxYZoomout: number;
        initialZoomout: number;
    };
    function newZoomInfo(minXZoomout: number, maxXZoomout: number, minYZoomout: number, maxYZoomout: number, initialZoomout: number): ZoomInfo;
    function zoomInfoZoomOut(zoom: ZoomInfo, ratio: number): void;
    function zoomInfoSetZoomout(zoom: ZoomInfo, newZoomout: number): void;
    function zoomAllZoomout(world: Rectangle, screen: Rectangle): number;
    function centeredVisWorld(world: Rectangle, screen: Rectangle, zoom: ZoomInfo): Rectangle;
    function constrain(value: number, min: number, max: number): number;
    function arcPathD_circle(endpoints: Line, maxDeviation: number, smartDeviationParam: number, invert: boolean, x2yZoomRatio?: number): string;
    function arcPathD_ellipse(endpoints: Line, worldWidth: number, maxMinor: number, invert: boolean, x2yZoomRatio?: number): string;
    function arcPathD_bezier(endpoints: Line, maxDeviation: number, smartDeviationParam: number, invert: boolean, x2yZoomRatio?: number): string;
    function symCdfPolygonPoints(boundingBox: Rectangle, cdf: number[][]): string;
    function rectToScreen(visWorld: Rectangle, screen: Rectangle, rect: Rectangle): Rectangle;
    function lineToScreen(visWorld: Rectangle, screen: Rectangle, line: Line): Line;
    function round(num: number, ndigits?: number): number;
}
