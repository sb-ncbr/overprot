import { Constants } from './Constants';
export var Geometry;
(function (Geometry) {
    function newRectangle() {
        return { x: -1, y: -1, width: 2, height: 2 };
    }
    Geometry.newRectangle = newRectangle;
    function rectangleFromCanvas(canvas) {
        let width = +canvas.style('width').replace('px', '');
        let height = +canvas.style('height').replace('px', '');
        return { x: 0, y: 0, width: width, height: height };
    }
    Geometry.rectangleFromCanvas = rectangleFromCanvas;
    function rectangleCenter(rect) {
        return [rect.x + 0.5 * rect.width, rect.y + 0.5 * rect.height];
    }
    Geometry.rectangleCenter = rectangleCenter;
    function newZoomInfo(minXZoomout, maxXZoomout, minYZoomout, maxYZoomout, initialZoomout) {
        if (minXZoomout > maxXZoomout)
            console.warn('newZoomInfo(): minXZoomout > maxXZoomout');
        if (minYZoomout > maxYZoomout)
            console.warn('newZoomInfo(): minYZoomout > maxYZoomout');
        let minZoomout = Math.min(minXZoomout, minYZoomout);
        let maxZoomout = Math.max(maxXZoomout, maxYZoomout);
        let zoomout = constrain(initialZoomout, minZoomout, maxXZoomout);
        let result = {
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
    Geometry.newZoomInfo = newZoomInfo;
    function zoomInfoZoomOut(zoom, ratio) {
        zoomInfoSetZoomout(zoom, zoom.zoomout * ratio);
    }
    Geometry.zoomInfoZoomOut = zoomInfoZoomOut;
    function zoomInfoSetZoomout(zoom, newZoomout) {
        zoom.zoomout = constrain(newZoomout, zoom.minZoomout, zoom.maxZoomout);
        zoom.xZoomout = constrain(zoom.zoomout, zoom.minXZoomout, zoom.maxXZoomout);
        zoom.yZoomout = constrain(zoom.zoomout, zoom.minYZoomout, zoom.maxYZoomout);
    }
    Geometry.zoomInfoSetZoomout = zoomInfoSetZoomout;
    function zoomAllZoomout(world, screen) {
        return Math.max(world.width / screen.width, world.height / screen.height);
    }
    Geometry.zoomAllZoomout = zoomAllZoomout;
    function centeredVisWorld(world, screen, zoom) {
        let visWidth = screen.width * zoom.xZoomout;
        let visHeight = screen.height * zoom.yZoomout;
        // let worldCenterX = world.x + 0.5 * world.width;
        // let worldCenterY = world.y + 0.5 * world.height;
        let [worldCenterX, worldCenterY] = rectangleCenter(world);
        return { x: worldCenterX - 0.5 * visWidth, y: worldCenterY - 0.5 * visHeight, width: visWidth, height: visHeight };
    }
    Geometry.centeredVisWorld = centeredVisWorld;
    function constrain(value, min, max) {
        return Math.min(Math.max(value, min), max);
    }
    Geometry.constrain = constrain;
    function arcPathD_circle(endpoints, maxDeviation, smartDeviationParam, invert, x2yZoomRatio = 1) {
        if (maxDeviation == 0) {
            return `M${endpoints.x1},${endpoints.y1} L${endpoints.x2},${endpoints.y2}`;
        }
        else {
            let { x1, y1, x2, y2 } = endpoints;
            let dx = x2 - x1;
            let dy = y2 - y1;
            let xDistance = Math.abs(dx);
            let deviation = maxDeviation * xDistance / (xDistance + smartDeviationParam); // deviation from straight line if x weren't distorted
            let weightedSqDistance = Math.pow((dx / x2yZoomRatio), 2) + Math.pow(dy, 2); // distance if x weren't distorted
            let radius = deviation / 2 + weightedSqDistance / (8 * deviation); // circle radius if x weren't distorted
            let xRadius = radius * x2yZoomRatio;
            return `M${x2},${y2} A${xRadius},${radius} 0 0,${invert ? 1 : 0} ${x1},${y1}`;
        }
    }
    Geometry.arcPathD_circle = arcPathD_circle;
    function arcPathD_ellipse(endpoints, worldWidth, maxMinor, invert, x2yZoomRatio = 1) {
        if (maxMinor == 0) {
            return `M${endpoints.x1},${endpoints.y1} L${endpoints.x2},${endpoints.y2}`;
        }
        else {
            let { x1, y1, x2, y2 } = endpoints;
            let dy = y2 - y1;
            let dxStretched = x2 - x1;
            let dxOrig = dxStretched / x2yZoomRatio;
            let major = Math.sqrt(Math.pow(dxOrig, 2) + Math.pow(dy, 2)) / 2 + Constants.ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH * worldWidth / x2yZoomRatio;
            let minor = Math.min(maxMinor * Math.sqrt(Math.abs(dxStretched) / worldWidth), major);
            let theta = Math.atan(dy / dxOrig);
            let [major_, minor_, theta_] = deformEllipse(major, minor, theta, x2yZoomRatio);
            theta_ *= 180 / Math.PI; // radians -> degrees
            return `M${x2},${y2} A${major_},${minor_} ${theta_} 0,${invert ? 1 : 0} ${x1},${y1}`;
        }
    }
    Geometry.arcPathD_ellipse = arcPathD_ellipse;
    function arcPathD_bezier(endpoints, maxDeviation, smartDeviationParam, invert, x2yZoomRatio = 1) {
        if (maxDeviation == 0) {
            return `M${endpoints.x1},${endpoints.y1} L${endpoints.x2},${endpoints.y2}`;
        }
        else {
            let { x1, y1, x2, y2 } = endpoints;
            let dx = x2 - x1;
            let dy = y2 - y1;
            let xDistance = Math.abs(dx);
            let deviation = maxDeviation * xDistance / (xDistance + smartDeviationParam) * (invert ? -1 : 1);
            const SKEW = 0.05 * dx;
            const P = 1.32;
            return `M${x1},${y1} C${x1 + SKEW},${y1 - deviation * P} ${x2 - SKEW},${y2 - deviation * P} ${x2},${y2}`;
        }
    }
    Geometry.arcPathD_bezier = arcPathD_bezier;
    /** Scale ellipse along x-axis and return the major semiaxis a', minor semiaxis b', and angle theta' of the new ellipse (angles are in radians) */
    function deformEllipse_old(a, b, theta, q) {
        const { cos, sin, sqrt } = Math;
        if (q == 1) {
            return [a, b, theta];
        }
        if (sin(theta) == 0) { // ellipse is "lying"
            return [q * a, b, theta];
        }
        if (cos(theta) == 0) { // ellipse is "standing"
            return [q * a, b, theta];
        }
        const A = 2 * a * b * (Math.pow(q, 2) - 1) * cos(theta) * sin(theta);
        const B = 2 * (Math.pow(q, 2) * Math.pow(b, 2) - Math.pow(a, 2)) * Math.pow(sin(theta), 2) + 2 * (Math.pow(b, 2) - Math.pow(q, 2) * Math.pow(a, 2)) * Math.pow(cos(theta), 2);
        const D = Math.pow(B, 2) + 4 * Math.pow(A, 2); // because C = -A
        const t1 = (-B + sqrt(D)) / (2 * A); // t = tan(phi), where phi is the old ellipse phase for which the new ellipse is farthest/nearest to the center
        const t2 = (-B - sqrt(D)) / (2 * A);
        const r_1sq = ((Math.pow(q, 2) * Math.pow(a, 2) * Math.pow(cos(theta), 2) + Math.pow(a, 2) * Math.pow(sin(theta), 2)) * Math.pow(t1, 2)
            + (2 * a * b - 2 * Math.pow(q, 2) * a * b) * cos(theta) * sin(theta) * t1
            + (Math.pow(q, 2) * Math.pow(b, 2) * Math.pow(sin(theta), 2) + Math.pow(b, 2) * Math.pow(cos(theta), 2))) / (1 + Math.pow(t1, 2));
        const r_2sq = ((Math.pow(q, 2) * Math.pow(a, 2) * Math.pow(cos(theta), 2) + Math.pow(a, 2) * Math.pow(sin(theta), 2)) * Math.pow(t2, 2)
            + (2 * a * b - 2 * Math.pow(q, 2) * a * b) * cos(theta) * sin(theta) * t2
            + (Math.pow(q, 2) * Math.pow(b, 2) * Math.pow(sin(theta), 2) + Math.pow(b, 2) * Math.pow(cos(theta), 2))) / (1 + Math.pow(t2, 2));
        const a_ = sqrt(Math.max(r_1sq, r_2sq));
        const b_ = sqrt(Math.min(r_1sq, r_2sq));
        const sin2theta_ = sin(2 * theta) * (1 / Math.pow(a, 2) - 1 / Math.pow(b, 2)) / q / (1 / Math.pow(a_, 2) - 1 / Math.pow(b_, 2));
        const theta_ = 0.5 * Math.asin(sin2theta_);
        console.log(a, b, theta, '-->', a_, b_, theta_);
        return [a_, b_, theta_];
    }
    /** Scale ellipse along x-axis and return the major semiaxis a', minor semiaxis b', and angle theta' of the new ellipse (angles are in radians) */
    function deformEllipse(a, b, theta, q) {
        const { cos, sin, sqrt, atan } = Math;
        if (q == 1) {
            return [a, b, theta];
        }
        if (theta == 0 || sin(2 * theta) == 0) { // ellipse is "lying" or "standing"
            return [q * a, b, theta];
        }
        if (a == b) {
            return [q * a, b, 0];
        }
        const K1 = Math.pow(q, 2) * Math.pow(a, 2) * Math.pow(cos(theta), 2) + Math.pow(a, 2) * Math.pow(sin(theta), 2);
        const K2 = 2 * (1 - Math.pow(q, 2)) * a * b * cos(theta) * sin(theta);
        const K3 = Math.pow(q, 2) * Math.pow(b, 2) * Math.pow(sin(theta), 2) + Math.pow(b, 2) * Math.pow(cos(theta), 2);
        const A = K2;
        const B = 2 * (K1 - K3);
        const C = -K2;
        const D = Math.pow(B, 2) - 4 * A * C;
        let t1 = (-B + sqrt(D)) / (2 * A);
        let t2 = (-B - sqrt(D)) / (2 * A);
        let a_ = sqrt((K1 + K2 * t1 + K3 * Math.pow(t1, 2)) / (1 + Math.pow(t1, 2)));
        let b_ = sqrt((K1 + K2 * t2 + K3 * Math.pow(t2, 2)) / (1 + Math.pow(t2, 2)));
        if (a_ < b_) {
            [t1, t2] = [t2, t1];
            [a_, b_] = [b_, a_];
        }
        const sqcosphi = 1 / (1 + Math.pow(t1, 2));
        const sqsinphi = 1 - sqcosphi;
        const cossinphi = t1 / (1 + Math.pow(t1, 2));
        const numer = Math.pow(a, 2) * cos(theta) * sin(theta) * sqcosphi + a * b * cossinphi + Math.pow(b, 2) * cos(theta) * sin(theta) * sqsinphi;
        const denom = q * (Math.pow(a, 2) * Math.pow(cos(theta), 2) * sqcosphi - Math.pow(b, 2) * Math.pow(sin(theta), 2) * sqsinphi);
        let theta_ = atan(numer / denom);
        return [a_, b_, theta_];
    }
    function symCdfPolygonPoints(boundingBox, cdf) {
        let [x0, y0] = rectangleCenter(boundingBox);
        let maxLength = cdf[cdf.length - 1][0];
        let occurrence = cdf[0][0] > 0 ? 1 : 1 - cdf[0][1];
        let xScale = 0.5 * boundingBox.width / maxLength;
        let yScale = 0.5 * boundingBox.height / occurrence;
        // Staircase variant:
        let relPoints = [];
        if (cdf[0][0] > 0) {
            relPoints.push([0, yScale]);
            relPoints.push([cdf[0][0] * xScale, yScale]);
        }
        relPoints.push([cdf[0][0] * xScale, (1 - cdf[0][1]) * yScale]);
        for (let i = 1; i < cdf.length; i++) {
            relPoints.push([cdf[i][0] * xScale, (1 - cdf[i - 1][1]) * yScale]);
            relPoints.push([cdf[i][0] * xScale, (1 - cdf[i][1]) * yScale]);
        }
        let points = [];
        for (let i = 0; i < relPoints.length - 1; i++)
            points.push([x0 + relPoints[i][0], y0 + relPoints[i][1]]);
        for (let i = relPoints.length - 1; i > 0; i--)
            points.push([x0 + relPoints[i][0], y0 - relPoints[i][1]]);
        for (let i = 0; i < relPoints.length - 1; i++)
            points.push([x0 - relPoints[i][0], y0 - relPoints[i][1]]);
        for (let i = relPoints.length - 1; i > 0; i--)
            points.push([x0 - relPoints[i][0], y0 + relPoints[i][1]]);
        // // Smooth variant
        // let points: number[][] = [];
        // for (let i = 0; i < cdf.length-1; i++) points.push([x0 + cdf[i][0] * xScale, y0 + (1 - cdf[i][1]) * yScale]);
        // for (let i = cdf.length-1; i > 0; i--) points.push([x0 + cdf[i][0] * xScale, y0 - (1 - cdf[i][1]) * yScale]);
        // for (let i = 0; i < cdf.length-1; i++) points.push([x0 - cdf[i][0] * xScale, y0 - (1 - cdf[i][1]) * yScale]);
        // for (let i = cdf.length-1; i > 0; i--) points.push([x0 - cdf[i][0] * xScale, y0 + (1 - cdf[i][1]) * yScale]);
        return points.map(xy => round(xy[0], 1) + ',' + round(xy[1], 1)).join(' ');
    }
    Geometry.symCdfPolygonPoints = symCdfPolygonPoints;
    function xyToScreen(visWorld, screen, x, y) {
        let xScreen = (x - visWorld.x) / visWorld.width * screen.width + screen.x;
        let yScreen = (y - visWorld.y) / visWorld.height * screen.height + screen.y;
        return [xScreen, yScreen];
    }
    function rectToScreen(visWorld, screen, rect) {
        let { x, y, width: w, height: h } = rect;
        let xScreen = (x - visWorld.x) / visWorld.width * screen.width + screen.x;
        let yScreen = (y - visWorld.y) / visWorld.height * screen.height + screen.y;
        let width = w / visWorld.width * screen.width;
        let height = h / visWorld.height * screen.height;
        return { x: xScreen, y: yScreen, width: width, height: height };
    }
    Geometry.rectToScreen = rectToScreen;
    function lineToScreen(visWorld, screen, line) {
        let { x1, y1, x2, y2 } = line;
        let x1Screen = (x1 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y1Screen = (y1 - visWorld.y) / visWorld.height * screen.height + screen.y;
        let x2Screen = (x2 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y2Screen = (y2 - visWorld.y) / visWorld.height * screen.height + screen.y;
        return { x1: x1Screen, y1: y1Screen, x2: x2Screen, y2: y2Screen };
    }
    Geometry.lineToScreen = lineToScreen;
    function lineToScreen2(visWorld, screen, line) {
        let { x1, y1, x2, y2 } = line;
        let x1Screen = (x1 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y1Screen = (y1 - visWorld.y) / visWorld.height * screen.height + screen.y;
        let x2Screen = (x2 - visWorld.x) / visWorld.width * screen.width + screen.x;
        let y2Screen = (y2 - visWorld.y) / visWorld.height * screen.height + screen.y;
        return { x1: x1Screen, y1: y1Screen, x2: x2Screen, y2: y2Screen - 20 };
    }
    function polylineToScreen(visWorld, screen, polyline) {
        return polyline.map(xy => [
            (xy[0] - visWorld.x) / visWorld.width * screen.width + screen.x,
            (xy[1] - visWorld.y) / visWorld.height * screen.height + screen.y,
        ]);
    }
    function round(num, ndigits = 0) {
        return Math.round(num * Math.pow(10, ndigits)) / Math.pow(10, ndigits);
    }
    Geometry.round = round;
})(Geometry || (Geometry = {}));
//# sourceMappingURL=Geometry.js.map