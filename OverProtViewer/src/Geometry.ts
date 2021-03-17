import * as d3 from 'd3';
import { min } from 'd3';
import { Constants } from './Constants';


export namespace Geometry {

    export type Rectangle = { x: number, y: number, width: number, height: number };
    export function newRectangle(): Rectangle {
        return { x: -1, y: -1, width: 2, height: 2 };
    }
    export function rectangleFromCanvas(canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>): Rectangle {
        let width = + canvas.style('width').replace('px', '');
        let height = + canvas.style('height').replace('px', '');
        return { x: 0, y: 0, width: width, height: height };
    }
    export function rectangleCenter(rect: Rectangle): [number, number] {
        return [rect.x + 0.5*rect.width, rect.y + 0.5*rect.height];
    }

    export type Line = { x1: number, y1: number, x2: number, y2: number };

    export type Polyline = number[][];

    export type ZoomInfo = {
        zoomout: number,
        xZoomout: number,
        yZoomout: number,
        minZoomout: number,
        maxZoomout: number,
        minXZoomout: number,
        maxXZoomout: number,
        minYZoomout: number,
        maxYZoomout: number,
        initialZoomout: number,
    }


    export function newZoomInfo(minXZoomout: number, maxXZoomout: number, minYZoomout: number, maxYZoomout: number, initialZoomout: number): ZoomInfo {
        if (minXZoomout > maxXZoomout) console.warn('newZoomInfo(): minXZoomout > maxXZoomout');
        if (minYZoomout > maxYZoomout) console.warn('newZoomInfo(): minYZoomout > maxYZoomout');
        let minZoomout = Math.min(minXZoomout, minYZoomout);
        let maxZoomout = Math.max(maxXZoomout, maxYZoomout);
        let zoomout = constrain(initialZoomout, minZoomout, maxXZoomout);
        let result: ZoomInfo = {
            zoomout: zoomout,
            xZoomout: constrain(zoomout, minXZoomout, maxXZoomout),
            yZoomout: constrain(zoomout, minYZoomout, maxYZoomout),
            minZoomout: minZoomout,
            maxZoomout: maxZoomout,
            minXZoomout: minXZoomout,
            maxXZoomout: maxXZoomout,
            minYZoomout: minYZoomout,
            maxYZoomout: maxYZoomout,
            initialZoomout: initialZoomout,
        };
        return result;
    }
    export function zoomInfoZoomOut(zoom: ZoomInfo, ratio: number): void {
        zoomInfoSetZoomout(zoom, zoom.zoomout * ratio);
    }
    export function zoomInfoSetZoomout(zoom: ZoomInfo, newZoomout: number): void {
        zoom.zoomout = constrain(newZoomout, zoom.minZoomout, zoom.maxZoomout);
        zoom.xZoomout = constrain(zoom.zoomout, zoom.minXZoomout, zoom.maxXZoomout);
        zoom.yZoomout = constrain(zoom.zoomout, zoom.minYZoomout, zoom.maxYZoomout);
    }
    export function zoomAllZoomout(world: Rectangle, screen: Rectangle): number {
        return Math.max(world.width / screen.width, world.height / screen.height);
    }
    export function centeredVisWorld(world: Rectangle, screen: Rectangle, zoom: ZoomInfo): Rectangle {
        let visWidth = screen.width * zoom.xZoomout;
        let visHeight = screen.height * zoom.yZoomout;
        // let worldCenterX = world.x + 0.5 * world.width;
        // let worldCenterY = world.y + 0.5 * world.height;
        let [worldCenterX, worldCenterY] = rectangleCenter(world);
        return { x: worldCenterX - 0.5 * visWidth, y: worldCenterY - 0.5 * visHeight, width: visWidth, height: visHeight };
    }

    export function constrain(value: number, min: number, max: number): number {
        return Math.min(Math.max(value, min), max);
    }


    export function arcPathD_circle(endpoints: Line, maxDeviation: number, smartDeviationParam: number, invert: boolean, x2yZoomRatio: number = 1): string {
        if (maxDeviation == 0) {
            return `M${endpoints.x1},${endpoints.y1} L${endpoints.x2},${endpoints.y2}`;
        } else {
            let { x1, y1, x2, y2 } = endpoints;
            let dx = x2 - x1;
            let dy = y2 - y1;
            let xDistance = Math.abs(dx);
            let deviation = maxDeviation * xDistance / (xDistance + smartDeviationParam);  // deviation from straight line if x weren't distorted
            let weightedSqDistance = (dx/x2yZoomRatio)**2 + dy**2;  // distance if x weren't distorted
            let radius = deviation / 2 + weightedSqDistance / (8 * deviation);  // circle radius if x weren't distorted
            let xRadius = radius * x2yZoomRatio;
            return `M${x2},${y2} A${xRadius},${radius} 0 0,${invert ? 1 : 0} ${x1},${y1}`;
        }
    }
    export function arcPathD_ellipse(endpoints: Line, worldWidth: number, maxMinor: number, invert: boolean, x2yZoomRatio: number = 1): string {
        if (maxMinor == 0) {
            return `M${endpoints.x1},${endpoints.y1} L${endpoints.x2},${endpoints.y2}`;
        } else {
            let { x1, y1, x2, y2 } = endpoints;
            let dy = y2 - y1;
            let dxStretched = x2 - x1;
            let dxOrig = dxStretched / x2yZoomRatio;
            let major = Math.sqrt(dxOrig**2 + dy**2) / 2 + Constants.ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH * worldWidth / x2yZoomRatio;
            let minor = Math.min(maxMinor * Math.sqrt(Math.abs(dxStretched) / worldWidth), major);
            let theta = Math.atan(dy / dxOrig);
            let [major_, minor_, theta_] = deformEllipse(major, minor, theta, x2yZoomRatio);
            theta_ *= 180 / Math.PI;  // radians -> degrees
            return `M${x2},${y2} A${major_},${minor_} ${theta_} 0,${invert ? 1 : 0} ${x1},${y1}`;
        }
    }
    export function arcPathD_bezier(endpoints: Line, maxDeviation: number, smartDeviationParam: number, invert: boolean, x2yZoomRatio: number = 1): string {
        if (maxDeviation == 0) {
            return `M${endpoints.x1},${endpoints.y1} L${endpoints.x2},${endpoints.y2}`;
        } else {
            let { x1, y1, x2, y2 } = endpoints;
            let dx = x2 - x1;
            let dy = y2 - y1;
            let xDistance = Math.abs(dx);
            let deviation = maxDeviation * xDistance / (xDistance + smartDeviationParam) * (invert?-1:1);
            const SKEW = 0.05 * dx;
            const P = 1.32;
            return `M${x1},${y1} C${x1+SKEW},${y1-deviation*P} ${x2-SKEW},${y2-deviation*P} ${x2},${y2}`
        }
    }

    /** Scale ellipse along x-axis and return the major semiaxis a', minor semiaxis b', and angle theta' of the new ellipse (angles are in radians) */
    function deformEllipse_old(a: number, b: number, theta: number, q: number): [number, number, number]{
        const {cos, sin, sqrt} = Math;
        if (q == 1){
            return [a, b, theta];
        }
        if (sin(theta) == 0){  // ellipse is "lying"
            return [q * a, b, theta];
        }
        if (cos(theta) == 0){  // ellipse is "standing"
            return [q * a, b, theta];
        }
        const A = 2 * a * b * (q**2 - 1) * cos(theta) * sin(theta);
        const B = 2 * (q**2 * b**2 - a**2) * sin(theta)**2 + 2 * (b**2 - q**2 * a**2) * cos(theta)**2;
        const D = B**2 + 4 * A**2;  // because C = -A
        const t1 = (-B + sqrt(D)) / (2 * A);  // t = tan(phi), where phi is the old ellipse phase for which the new ellipse is farthest/nearest to the center
        const t2 = (-B - sqrt(D)) / (2 * A);
        const r_1sq = ( (q**2 * a**2 * cos(theta)**2 + a**2 * sin(theta)**2) * t1**2
                       +(2*a*b - 2*q**2*a*b) * cos(theta) * sin(theta) * t1
                       +(q**2 * b**2 * sin(theta)**2 + b**2 * cos(theta)**2) ) / (1 + t1**2);
        const r_2sq = ( (q**2 * a**2 * cos(theta)**2 + a**2 * sin(theta)**2) * t2**2
                        +(2*a*b - 2*q**2*a*b) * cos(theta) * sin(theta) * t2
                        +(q**2 * b**2 * sin(theta)**2 + b**2 * cos(theta)**2) ) / (1 + t2**2);
        const a_ = sqrt(Math.max(r_1sq, r_2sq));                        
        const b_ = sqrt(Math.min(r_1sq, r_2sq));  
        const sin2theta_ = sin(2 * theta) * (1/a**2 - 1/b**2) / q / (1/a_**2 - 1/b_**2);   
        const theta_ = 0.5 * Math.asin(sin2theta_);
        console.log(a, b, theta, '-->', a_, b_, theta_);
        return [a_, b_, theta_];
    }

    /** Scale ellipse along x-axis and return the major semiaxis a', minor semiaxis b', and angle theta' of the new ellipse (angles are in radians) */
    function deformEllipse(a: number, b: number, theta: number, q: number): [number, number, number]{
        const {cos, sin, sqrt, atan} = Math;
        if (q == 1) {
            return [a, b, theta];
        }
        if (theta == 0 || sin(2*theta) == 0) { // ellipse is "lying" or "standing"
            return [q * a, b, theta];
        }  
        if (a == b) {
            return [q*a, b, 0];
        }
        const K1 = q**2 * a**2 * cos(theta)**2 + a**2 * sin(theta)**2;
        const K2 = 2 * (1 - q**2) * a * b * cos(theta) * sin(theta);
        const K3 = q**2 * b**2 * sin(theta)**2 + b**2 * cos(theta)**2;
        const A = K2;
        const B = 2 * (K1 - K3);
        const C = -K2;
        const D = B**2 - 4 * A * C;
        let t1 = (-B + sqrt(D)) / (2 * A);
        let t2 = (-B - sqrt(D)) / (2 * A);
        let a_ = sqrt((K1 + K2 * t1 + K3 * t1**2) / (1 + t1 **2));
        let b_ = sqrt((K1 + K2 * t2 + K3 * t2**2) / (1 + t2 **2));
        if (a_ < b_) {
            [t1, t2] = [t2, t1];
            [a_, b_] = [b_, a_];
        }
        const sqcosphi = 1 / (1 + t1**2);
        const sqsinphi = 1 - sqcosphi;
        const cossinphi = t1 / (1 + t1**2);
        const numer = a**2 * cos(theta) * sin(theta) * sqcosphi + a * b * cossinphi + b**2 * cos(theta) * sin(theta) * sqsinphi;
        const denom = q * (a**2 * cos(theta)**2 * sqcosphi - b**2 * sin(theta)**2 * sqsinphi);
        let theta_ = atan(numer / denom);
        return [a_, b_, theta_];
    }

    export function symCdfPolygonPoints(boundingBox: Rectangle, cdf: number[][]): string {
        let [x0, y0] = rectangleCenter(boundingBox);
        let maxLength = cdf[cdf.length - 1][0];
        let occurrence = cdf[0][0] > 0 ? 1 : 1 - cdf[0][1];
        let xScale = 0.5 * boundingBox.width / maxLength;
        let yScale = 0.5 * boundingBox.height / occurrence;

        // Staircase variant:
        let relPoints: number[][] = [];
        if (cdf[0][0] > 0) {
            relPoints.push([0, yScale]);
            relPoints.push([cdf[0][0] * xScale, yScale]);
        }
        relPoints.push([cdf[0][0] * xScale, (1 - cdf[0][1]) * yScale]);
        for (let i = 1; i < cdf.length; i++) {
            relPoints.push([cdf[i][0] * xScale, (1 - cdf[i - 1][1]) * yScale]);
            relPoints.push([cdf[i][0] * xScale, (1 - cdf[i][1]) * yScale]);
        }
        let points: number[][] = [];
        for (let i = 0; i < relPoints.length - 1; i++) points.push([x0 + relPoints[i][0], y0 + relPoints[i][1]]);
        for (let i = relPoints.length - 1; i > 0; i--) points.push([x0 + relPoints[i][0], y0 - relPoints[i][1]]);
        for (let i = 0; i < relPoints.length - 1; i++) points.push([x0 - relPoints[i][0], y0 - relPoints[i][1]]);
        for (let i = relPoints.length - 1; i > 0; i--) points.push([x0 - relPoints[i][0], y0 + relPoints[i][1]]);

        // // Smooth variant
        // let points: number[][] = [];
        // for (let i = 0; i < cdf.length-1; i++) points.push([x0 + cdf[i][0] * xScale, y0 + (1 - cdf[i][1]) * yScale]);
        // for (let i = cdf.length-1; i > 0; i--) points.push([x0 + cdf[i][0] * xScale, y0 - (1 - cdf[i][1]) * yScale]);
        // for (let i = 0; i < cdf.length-1; i++) points.push([x0 - cdf[i][0] * xScale, y0 - (1 - cdf[i][1]) * yScale]);
        // for (let i = cdf.length-1; i > 0; i--) points.push([x0 - cdf[i][0] * xScale, y0 + (1 - cdf[i][1]) * yScale]);

        return points.map(xy => round(xy[0], 1) + ',' + round(xy[1], 1)).join(' ');
    }

    function xyToScreen(visWorld: Rectangle, screen: Rectangle, x: number, y: number): number[] {
        let xScreen = (x - visWorld.x) / visWorld.width * screen.width + screen.x;
        let yScreen = (y - visWorld.y) / visWorld.height * screen.height + screen.y;
        return [xScreen, yScreen];
    }

    export function rectToScreen(visWorld: Rectangle, screen: Rectangle, rect: Rectangle): Rectangle {
        let { x, y, width: w, height: h } = rect;
        let xScreen = (x - visWorld.x) / visWorld.width * screen.width + screen.x;
        let yScreen = (y - visWorld.y) / visWorld.height * screen.height + screen.y;
        let width = w / visWorld.width * screen.width;
        let height = h / visWorld.height * screen.height;
        return { x: xScreen, y: yScreen, width: width, height: height };
    }

    export function lineToScreen(visWorld: Rectangle, screen: Rectangle, line: Line): Line {
        let { x1, y1, x2, y2 } = line;
        let x1Screen = (x1 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y1Screen = (y1 - visWorld.y) / visWorld.height * screen.height + screen.y;
        let x2Screen = (x2 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y2Screen = (y2 - visWorld.y) / visWorld.height * screen.height + screen.y;
        return { x1: x1Screen, y1: y1Screen, x2: x2Screen, y2: y2Screen };
    }
    function lineToScreen2(visWorld: Rectangle, screen: Rectangle, line: Line): Line {
        let { x1, y1, x2, y2 } = line;
        let x1Screen = (x1 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y1Screen = (y1 - visWorld.y) / visWorld.height * screen.height + screen.y;
        let x2Screen = (x2 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y2Screen = (y2 - visWorld.y) / visWorld.height * screen.height + screen.y;
        return { x1: x1Screen, y1: y1Screen, x2: x2Screen, y2: y2Screen - 20 };
    }

    function polylineToScreen(visWorld: Rectangle, screen: Rectangle, polyline: Polyline): Polyline {
        return polyline.map(xy => [
            (xy[0] - visWorld.x) / visWorld.width * screen.width + screen.x,
            (xy[1] - visWorld.y) / visWorld.height * screen.height + screen.y,
        ]);
    }

    export function round(num: number, ndigits = 0): number {
        return Math.round(num * 10 ** ndigits) / 10 ** ndigits;
    }

}