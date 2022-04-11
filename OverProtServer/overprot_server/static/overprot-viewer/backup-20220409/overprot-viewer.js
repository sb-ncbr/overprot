(function (d3) {
    'use strict';

    var Colors;
    (function (Colors) {
        var _a;
        _a = makeScheme(), Colors.NEUTRAL_COLOR = _a.neutral, Colors.COLOR_SCHEME = _a.scheme;
        Colors.NEUTRAL_DARK = d3.rgb(0, 0, 0);
        function makeScheme() {
            let scheme = d3.schemeCategory10.map(str => d3.rgb(str));
            let neutral = scheme.splice(7, 1);
            return { neutral: neutral[0], scheme: scheme };
        }
        function bySseType(type) {
            type = type.toLowerCase();
            if (type == 'h')
                return Colors.NEUTRAL_COLOR;
            else if (type == 'e')
                return Colors.COLOR_SCHEME[0];
            else
                return Colors.COLOR_SCHEME[1];
        }
        Colors.bySseType = bySseType;
        function byIndex0(i) {
            return Colors.COLOR_SCHEME[i % Colors.COLOR_SCHEME.length];
        }
        Colors.byIndex0 = byIndex0;
        function byIndex1(i) {
            if (i >= 1)
                return Colors.COLOR_SCHEME[(i - 1) % Colors.COLOR_SCHEME.length];
            else
                return Colors.NEUTRAL_COLOR;
        }
        Colors.byIndex1 = byIndex1;
        /** Maps 0 to the coolest, max to the hottest. */
        function byLinHeatmap(value, max) {
            const x = value / max;
            return d3.rgb(d3.interpolateInferno(x));
        }
        Colors.byLinHeatmap = byLinHeatmap;
        /** Maps 0 to the coolest, middle to the middle of the scale, infinity to the hottest. */
        function byExpHeatmap(value, middle) {
            const x = 1 - Math.pow(2, (-value / middle));
            return d3.rgb(d3.interpolateInferno(x));
        }
        Colors.byExpHeatmap = byExpHeatmap;
    })(Colors || (Colors = {}));

    var Enums;
    (function (Enums) {
        let ColorMethod;
        (function (ColorMethod) {
            ColorMethod[ColorMethod["Uniform"] = 0] = "Uniform";
            ColorMethod[ColorMethod["Type"] = 1] = "Type";
            ColorMethod[ColorMethod["Sheet"] = 2] = "Sheet";
            ColorMethod[ColorMethod["Stdev"] = 3] = "Stdev";
            ColorMethod[ColorMethod["Rainbow"] = 4] = "Rainbow";
        })(ColorMethod = Enums.ColorMethod || (Enums.ColorMethod = {}));
        let ShapeMethod;
        (function (ShapeMethod) {
            ShapeMethod[ShapeMethod["Rectangle"] = 0] = "Rectangle";
            ShapeMethod[ShapeMethod["SymCdf"] = 1] = "SymCdf";
        })(ShapeMethod = Enums.ShapeMethod || (Enums.ShapeMethod = {}));
        let LayoutMethod;
        (function (LayoutMethod) {
            LayoutMethod[LayoutMethod["Old"] = 0] = "Old";
            LayoutMethod[LayoutMethod["New"] = 1] = "New";
        })(LayoutMethod = Enums.LayoutMethod || (Enums.LayoutMethod = {}));
    })(Enums || (Enums = {}));

    var Constants;
    (function (Constants) {
        Constants.CANVAS_HEIGHT = 300;
        Constants.CANVAS_WIDTH = 1200;
        Constants.ZOOM_STEP_RATIO = 1.25;
        Constants.ZOOM_STEP_RATIO_MOUSE = Constants.ZOOM_STEP_RATIO;
        Constants.SHIFT_STEP_RELATIVE = 0.2;
        Constants.MAX_EMPTY_X_MARGIN = 0.0; // relative to screen
        Constants.MAX_EMPTY_Y_MARGIN = 0.0; // relative to screen
        Constants.MAX_X_ZOOMOUT = 1 / (1 - 2 * Constants.MAX_EMPTY_X_MARGIN);
        Constants.MAX_Y_ZOOMOUT = 1 / (1 - 2 * Constants.MAX_EMPTY_Y_MARGIN);
        Constants.TRANSITION_DURATION = d3.transition().duration(); //ms
        Constants.TOOLTIP_DELAY = 600; //ms
        Constants.MOUSE_HOLD_BEHAVIOR_INITIAL_SLEEP_TIME = 500; //ms
        Constants.MOUSE_HOLD_BEHAVIOR_STEP_SLEEP_TIME = 50; //ms
        Constants.TEMPORARY_CENTRAL_MESSAGE_TIMEOUT = 1000; //ms
        Constants.TOOLTIP_OFFSET = { x: 12, y: 0 }; // position of tooltip relative to cursor
        Constants.NODE_STROKE_WIDTH = 1;
        Constants.NODE_STROKE_WIDTH_HIGHLIGHTED = 3;
        Constants.EDGE_COLOR = '#808080';
        Constants.NODE_FILL = Colors.NEUTRAL_COLOR;
        Constants.NODE_STROKE = Colors.NEUTRAL_COLOR.darker();
        Constants.NODE_LABEL_COLOR = 'black';
        Constants.MINIMAL_WIDTH_FOR_SSE_LABEL = 20;
        Constants.HEATMAP_MIDDLE_VALUE = 5;
        Constants.DEFAULT_OCCURRENCE_THRESHOLD = 0.2;
        Constants.DEFAULT_BETA_CONNECTIVITY_VISIBILITY = true;
        Constants.DEFAULT_LAYOUT_METHOD = Enums.LayoutMethod.New;
        Constants.DEFAULT_COLOR_METHOD = Enums.ColorMethod.Sheet;
        Constants.DEFAULT_SHAPE_METHOD = Enums.ShapeMethod.Rectangle;
        Constants.DEFAULT_DISPATCH_EVENTS = false;
        Constants.DEFAULT_LISTEN_EVENTS = false;
        //#region measurements in the world
        Constants.LENGTH_SCALE = 4; // width of 1 residue in the world
        Constants.OCCURRENCE_SCALE = 100; // height of occurrence 1.0 (100%) in the world
        Constants.FLOOR_HEIGHT = 1.25 * Constants.OCCURRENCE_SCALE; // 1.5 * OCCURRENCE_SCALE;
        Constants.TOP_MARGIN = 0.25 * Constants.OCCURRENCE_SCALE;
        Constants.BOTTOM_MARGIN = 0.25 * Constants.OCCURRENCE_SCALE;
        Constants.LEFT_MARGIN = 4 * Constants.LENGTH_SCALE;
        Constants.RIGHT_MARGIN = 4 * Constants.LENGTH_SCALE;
        Constants.GAP_LENGTH = 3 * Constants.LENGTH_SCALE;
        Constants.KNOB_LENGTH = 1 * Constants.LENGTH_SCALE;
        Constants.ARC_MAX_DEVIATION = 0.5 * Constants.OCCURRENCE_SCALE + Math.min(Constants.TOP_MARGIN, Constants.BOTTOM_MARGIN);
        Constants.ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH = 0.001; // slightly increasing ellipse major semiaxis provides smaller angle of arc ends
        Constants.ARC_MAX_MINOR = Constants.ARC_MAX_DEVIATION / (1 - Math.sqrt(1 - Math.pow((1 / (1 + 2 * Constants.ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH)), 2))); // for elliptical arcs with extra major
        Constants.ARC_SMART_DEVIATION_PARAM_WRT_WORLD_WIDTH = 0.2; // ARC_SMART_DEVIATION_PARAM = distance for which the arc deviation is 1/2*ARC_MAX_DEVIATION (for circular arcs)
        //#endregion
        Constants.HELIX_TYPE = 'h';
        Constants.STRAND_TYPE = 'e';
        Constants.HANGING_TEXT_OFFSET = 5;
        Constants.RESET_SYMBOL = '&#x27F3;';
        Constants.OPEN_POPUP_SYMBOL = ' &#x25BE;';
        Constants.EVENT_PREFIX = 'PDB.overprot.';
        // Outbound events (dispatched by the viewer):
        Constants.EVENT_TYPE_SELECT = 'select';
        Constants.EVENT_TYPE_HOVER = 'hover';
        // Inbound events (listened to by the viewer):
        Constants.EVENT_TYPE_DO_SELECT = 'do.select';
        Constants.EVENT_TYPE_DO_HOVER = 'do.hover';
        Constants.ICON_PLUS = '<svg viewBox="0 0 100 100"><path d="M25,45 H45 V25 H55 V45 H75 V55 H55 V75 H45 V55 H25 z"></path></svg>';
        Constants.ICON_MINUS = '<svg viewBox="0 0 100 100"><path d="M25,45 H75 V55 H25 z"></path></svg>';
        Constants.ICON_RESET = '<svg viewBox="0 0 100 100"><path d="M50,25 A25,25,0,1,0,75,50 H65 A15,15,0,1,1,50,35 V47 L70,31 L50,15 z"></path></svg>';
        Constants.ICON_CAMERA = '<svg viewBox="0 0 100 100"><path d="' +
            'M15,30 H32 L40,22 H60 L68,30 H85 V72 H15 z ' + // body
            'M34,50 A16,16,0,0,0,66,50 A16,16,0,0,0,34,50 z ' + // around lens
            'M40,50 A10,10,0,0,0,60,50 A10,10,0,0,0,40,50 z ' + // lens
            'M80,35 V41 H72 V35 z ' + // window
            '"></path></svg>';
    })(Constants || (Constants = {}));

    var Geometry;
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
                let xDistance = Math.abs(dx);
                let deviation = maxDeviation * xDistance / (xDistance + smartDeviationParam) * (invert ? -1 : 1);
                const SKEW = 0.05 * dx;
                const P = 1.32;
                return `M${x1},${y1} C${x1 + SKEW},${y1 - deviation * P} ${x2 - SKEW},${y2 - deviation * P} ${x2},${y2}`;
            }
        }
        Geometry.arcPathD_bezier = arcPathD_bezier;
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
        function round(num, ndigits = 0) {
            return Math.round(num * Math.pow(10, ndigits)) / Math.pow(10, ndigits);
        }
        Geometry.round = round;
    })(Geometry || (Geometry = {}));

    var Dag;
    (function (Dag) {
        function newDag() {
            return { nodes: [], precedence: [], beta_connectivity: [], levels: [], precedenceLines: [], activeNodes: [], origPrecedence: [], origBetaConnectivity: [], error: null };
        }
        Dag.newDag = newDag;
        function dagFromJson(json) {
            var _a;
            let error = null;
            try {
                let dag = JSON.parse(json);
                error = dag.error = (_a = dag.error) !== null && _a !== void 0 ? _a : null;
                dag.nodes.forEach(node => node.active = true);
                dag.levels = [];
                dag.precedenceLines = [];
                dag.activeNodes = d3.range(dag.nodes.length);
                dag.origPrecedence = dag.precedence;
                dag.origBetaConnectivity = dag.beta_connectivity;
                addLaddersToNodes(dag);
                return dag;
            }
            catch (ex) {
                error = error !== null && error !== void 0 ? error : `Failed to parse input file.`;
                console.warn(error, '\n', ex, '\n', json);
                return newDagWithError(error);
            }
        }
        Dag.dagFromJson = dagFromJson;
        function newDagWithError(error) {
            let dag = newDag();
            dag.error = error;
            return dag;
        }
        Dag.newDagWithError = newDagWithError;
        function newNodeVisual() {
            return { rect: Geometry.newRectangle(), fill: Constants.NODE_FILL.hex(), stroke: Constants.NODE_STROKE.hex() };
        }
        Dag.newNodeVisual = newNodeVisual;
        function filterDagAndAddLevels(dag, occurrence_threshold) {
            let nNodes = dag.nodes.length;
            for (let node of dag.nodes)
                node.active = node.occurrence >= occurrence_threshold && node.avg_length > 0;
            let selectedIndices = d3.range(nNodes).filter(i => dag.nodes[i].active);
            let { ins: before, outs: after } = getInAndOutNeighbors(nNodes, dag.origPrecedence);
            for (let i = 0; i < nNodes; i++) {
                if (!dag.nodes[i].active) { // removing node
                    for (let b of before[i]) {
                        for (let a of after[i]) {
                            if (!(after[b].includes(a))) {
                                after[b].push(a);
                                before[a].push(b);
                            }
                        }
                    }
                }
            }
            dag.activeNodes = selectedIndices;
            dag.precedence = [];
            for (let b of selectedIndices) {
                for (let a of after[b]) {
                    if (dag.nodes[a].active)
                        dag.precedence.push([b, a]);
                }
            }
            addLevels(dag);
            removeRedundantPrecedenceEdges(dag);
        }
        Dag.filterDagAndAddLevels = filterDagAndAddLevels;
        function addLevels(dag) {
            let levels = [];
            let todoNodes = new Set(dag.activeNodes);
            let todoEdges = dag.precedence;
            while (todoNodes.size > 0) {
                let minVertices = new Set(todoNodes);
                todoEdges.forEach(e => minVertices.delete(e[1])); //remove vertices which are preceded by another vertex
                todoEdges = todoEdges.filter(e => !minVertices.has(e[0]));
                minVertices.forEach(v => todoNodes.delete(v));
                levels.push(Array.from(minVertices));
            }
            dag.levels = levels;
        }
        function removeRedundantPrecedenceEdges(dag) {
            let nNodes = dag.nodes.length;
            let nodeLevel = new Array(nNodes).fill(-1); // -1 = inactive node
            dag.levels.forEach((level, iLevel) => level.forEach(iNode => nodeLevel[iNode] = iLevel));
            let newPrecedence = [];
            let { outs: after } = getInAndOutNeighbors(nNodes, dag.precedence);
            for (let iLevel = 0; iLevel < dag.levels.length; iLevel++) {
                const level = dag.levels[iLevel];
                level.forEach(iNode => {
                    after[iNode].forEach(a => {
                        if (nodeLevel[a] == iLevel + 1 || !existsTransitivePath(iNode, a, after, nodeLevel)) {
                            newPrecedence.push([iNode, a]);
                        }
                    });
                });
            }
            dag.precedence = newPrecedence;
        }
        function existsTransitivePath(iFrom, iTo, outNeighbors, nodeLevel) {
            // Finding path of length > 1 from iFrom to iTo by DFS:
            let todo = outNeighbors[iFrom].filter(v => v != iTo);
            while (todo.length > 0) {
                let v = todo.pop();
                if (nodeLevel[v] != -1 && nodeLevel[v] < nodeLevel[iTo]) { // -1 = inactive node
                    todo.push(...outNeighbors[v]);
                }
                else if (v == iTo) {
                    return true;
                }
            }
            return false;
        }
        function getInAndOutNeighbors(nNodes, edges) {
            let ins = new Array(nNodes);
            let outs = new Array(nNodes);
            for (let i = 0; i < nNodes; i++) {
                ins[i] = [];
                outs[i] = [];
            }
            edges.forEach(edge => {
                ins[edge[1]].push(edge[0]);
                outs[edge[0]].push(edge[1]);
            });
            return { ins: ins, outs: outs };
        }
        function addLaddersToNodes(dag) {
            var _a, _b;
            for (let iLadder = 0; iLadder < dag.beta_connectivity.length; iLadder++) {
                const ladder = dag.beta_connectivity[iLadder];
                const u = ladder[0];
                const v = ladder[1];
                if (dag.nodes[u].ladders == undefined)
                    dag.nodes[u].ladders = [];
                (_a = dag.nodes[u].ladders) === null || _a === void 0 ? void 0 : _a.push(iLadder);
                if (dag.nodes[v].ladders == undefined)
                    dag.nodes[v].ladders = [];
                (_b = dag.nodes[v].ladders) === null || _b === void 0 ? void 0 : _b.push(iLadder);
            }
        }
        function getNodeMinMaxLength(node) {
            let lengthLevels = node.cdf.map(lp => lp[0]).filter(l => l > 0);
            if (lengthLevels.length == 0) {
                return [0, 0];
            }
            else {
                return [lengthLevels[0], lengthLevels[lengthLevels.length - 1]];
            }
        }
        Dag.getNodeMinMaxLength = getNodeMinMaxLength;
    })(Dag || (Dag = {}));

    var Types;
    (function (Types) {
        function newViewer(id, internalId, d3viewer, d3mainDiv, d3guiDiv, d3canvas, settings = null) {
            return {
                id: id,
                internalId: internalId,
                d3viewer: d3viewer,
                mainDiv: d3mainDiv,
                guiDiv: d3guiDiv,
                canvas: d3canvas,
                data: Dag.newDag(),
                world: Geometry.newRectangle(),
                visWorld: Geometry.newRectangle(),
                screen: Geometry.rectangleFromCanvas(d3canvas),
                zoom: Geometry.newZoomInfo(1, 1, 1, 1, 1),
                settings: settings !== null && settings !== void 0 ? settings : newSettings(),
                nodeMap: new Map(),
                ladderMap: new TupleMap(),
            };
        }
        Types.newViewer = newViewer;
        function newSettings() {
            return {
                file: '',
                width: Constants.CANVAS_WIDTH,
                height: Constants.CANVAS_HEIGHT,
                colorMethod: Constants.DEFAULT_COLOR_METHOD,
                shapeMethod: Constants.DEFAULT_SHAPE_METHOD,
                layoutMethod: Constants.DEFAULT_LAYOUT_METHOD,
                betaConnectivityVisibility: Constants.DEFAULT_BETA_CONNECTIVITY_VISIBILITY,
                occurrenceThreshold: Constants.DEFAULT_OCCURRENCE_THRESHOLD,
                dispatchEvents: Constants.DEFAULT_DISPATCH_EVENTS,
                listenEvents: Constants.DEFAULT_LISTEN_EVENTS
            };
        }
        Types.newSettings = newSettings;
        function newSettingsFromHTMLElement(element) {
            var _a;
            let MANDATORY_ATTRIBUTES = ['file'];
            let ALLOWED_ATTRIBUTES = ['id', 'file', 'width', 'height',
                'color-method', 'shape-method', 'layout-method', 'beta-connectivity', 'occurrence-threshold',
                'dispatch-events', 'listen-events'];
            MANDATORY_ATTRIBUTES.forEach(attributeName => {
                if (!element.hasAttribute(attributeName)) {
                    console.error(`Missing attribute: "${attributeName}".`);
                    // throw `Missing attribute: "${attributeName}".`;
                }
            });
            for (let i = 0; i < element.attributes.length; i++) {
                let attributeName = element.attributes.item(i).name;
                if (!ALLOWED_ATTRIBUTES.includes(attributeName)) {
                    console.warn(`Unknown attribute: "${attributeName}"`);
                }
            }
            let d3element = d3.select(element);
            const colorMethodDictionary = {
                'uniform': Enums.ColorMethod.Uniform,
                'type': Enums.ColorMethod.Type,
                'sheet': Enums.ColorMethod.Sheet,
                'variability': Enums.ColorMethod.Stdev,
                'rainbow': Enums.ColorMethod.Rainbow,
            };
            const shapeMethodDictionary = {
                'rectangle': Enums.ShapeMethod.Rectangle,
                'symcdf': Enums.ShapeMethod.SymCdf,
            };
            const layoutMethodDictionary = {
                'old': Enums.LayoutMethod.Old,
                'new': Enums.LayoutMethod.New,
            };
            const booleanDictionary = {
                'on': true,
                'off': false,
                'true': true,
                'false': false,
                '1': true,
                '0': false,
            };
            return {
                file: (_a = d3element.attr('file')) !== null && _a !== void 0 ? _a : '',
                height: parseIntAttribute('height', d3element.attr('height'), Constants.CANVAS_HEIGHT),
                width: parseIntAttribute('width', d3element.attr('width'), Constants.CANVAS_WIDTH),
                colorMethod: parseEnumAttribute('color-method', d3element.attr('color-method'), colorMethodDictionary, Constants.DEFAULT_COLOR_METHOD),
                shapeMethod: parseEnumAttribute('shape-method', d3element.attr('shape-method'), shapeMethodDictionary, Constants.DEFAULT_SHAPE_METHOD),
                layoutMethod: parseEnumAttribute('layout-method', d3element.attr('layout-method'), layoutMethodDictionary, Constants.DEFAULT_LAYOUT_METHOD),
                betaConnectivityVisibility: parseEnumAttribute('beta-connectivity', d3element.attr('beta-connectivity'), booleanDictionary, Constants.DEFAULT_BETA_CONNECTIVITY_VISIBILITY),
                occurrenceThreshold: parseFloatAttribute('occurrence-threshold', d3element.attr('occurrence-threshold'), Constants.DEFAULT_OCCURRENCE_THRESHOLD, [0, 1], true),
                dispatchEvents: parseEnumAttribute('dispatch-events', d3element.attr('dispatch-events'), booleanDictionary, Constants.DEFAULT_DISPATCH_EVENTS),
                listenEvents: parseEnumAttribute('listen-events', d3element.attr('listen-events'), booleanDictionary, Constants.DEFAULT_LISTEN_EVENTS),
            };
        }
        Types.newSettingsFromHTMLElement = newSettingsFromHTMLElement;
        function parseEnumAttribute(attributeName, attributeValue, dict, defaultValue) {
            if (attributeValue === null) {
                return defaultValue;
            }
            else if (dict[attributeValue] !== undefined) {
                return dict[attributeValue];
            }
            else {
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Allowed values: ${Object.keys(dict)}.`);
                return defaultValue;
            }
        }
        function parseIntAttribute(attributeName, attributeValue, defaultValue, minMaxLimits = []) {
            if (attributeValue === null) {
                return defaultValue;
            }
            else {
                let value = parseInt(attributeValue);
                if (isNaN(value)) {
                    console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be an integer.`);
                    return defaultValue;
                }
                else if (minMaxLimits.length >= 2 && (value < minMaxLimits[0] || value > minMaxLimits[1])) {
                    console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be an integer between ${minMaxLimits[0]} and ${minMaxLimits[1]}.`);
                    return defaultValue;
                }
                else {
                    return value;
                }
            }
        }
        function parseFloatAttribute(attributeName, attributeValue, defaultValue, minMaxLimits = [], allowPercentage = false) {
            if (attributeValue === null) {
                return defaultValue;
            }
            else {
                let value = parseFloat(attributeValue);
                if (allowPercentage && attributeValue.includes('%')) {
                    value *= 0.01;
                }
                if (isNaN(value)) {
                    console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be a float.`);
                    return defaultValue;
                }
                else if (minMaxLimits.length >= 2 && (value < minMaxLimits[0] || value > minMaxLimits[1])) {
                    console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be a float between ${minMaxLimits[0]} and ${minMaxLimits[1]}.`);
                    return defaultValue;
                }
                else {
                    return value;
                }
            }
        }
        class TupleMap {
            constructor() {
                this.map = undefined;
                this.value = undefined;
            }
            get(key) {
                var _a;
                let currentTM = this;
                for (const k of key) {
                    currentTM = (_a = currentTM.map) === null || _a === void 0 ? void 0 : _a.get(k);
                    if (currentTM == undefined)
                        return undefined;
                }
                return currentTM.value;
            }
            set(key, value) {
                let currentTM = this;
                for (const k of key) {
                    if (!currentTM.map) {
                        currentTM.map = new Map();
                    }
                    if (!currentTM.map.has(k)) {
                        currentTM.map.set(k, new TupleMap());
                    }
                    currentTM = currentTM.map.get(k);
                }
                currentTM.value = value;
            }
            entries() {
                let outList = [];
                this.collectEntries([], outList);
                return outList;
            }
            collectEntries(prefix, outList) {
                if (this.value != undefined) {
                    outList.push([[...prefix], this.value]);
                }
                if (this.map != undefined) {
                    for (const [k, sub] of this.map.entries()) {
                        prefix.push(k);
                        sub.collectEntries(prefix, outList);
                        prefix.pop();
                    }
                }
            }
        }
        Types.TupleMap = TupleMap;
    })(Types || (Types = {}));

    var Graphs;
    (function (Graphs) {
        function addXY(xy1, xy2) {
            return { x: xy1.x + xy2.x, y: xy1.y + xy2.y };
        }
        function newDagEmpty() {
            return { levels: [], edges: [], vertices: [], in_neighbors: new Map() };
        }
        Graphs.newDagEmpty = newDagEmpty;
        function newDagFromPath(vertices) {
            let dag = { levels: vertices.map(v => [v]), edges: [], vertices: [...vertices], in_neighbors: new Map() };
            for (let i = 0; i < vertices.length - 1; i++) {
                dag.edges.push([vertices[i], vertices[i + 1]]);
            }
            addInNeighbors(dag);
            return dag;
        }
        Graphs.newDagFromPath = newDagFromPath;
        function newDagFromPrecedence(levels, edges) {
            let vertices = [];
            for (const level of levels)
                for (const v of level)
                    vertices.push(v);
            let vertexSet = new Set(vertices);
            let finalEdges = [];
            for (const e of edges)
                if (vertexSet.has(e[0]) && vertexSet.has(e[1]))
                    finalEdges.push(e);
            let dag = { levels: levels, edges: finalEdges, vertices: vertices, in_neighbors: new Map() };
            addInNeighbors(dag);
            return dag;
        }
        Graphs.newDagFromPrecedence = newDagFromPrecedence;
        function addInNeighbors(dag) {
            for (const v of dag.vertices)
                dag.in_neighbors.set(v, []);
            for (const [u, v] of dag.edges)
                dag.in_neighbors.get(v).push(u);
        }
        function getSlice(dag, levelFrom, levelTo) {
            let sliceLevels = dag.levels.slice(levelFrom, levelTo);
            let sliceEdges = [];
            for (let iLevel = 1; iLevel < sliceLevels.length; iLevel++) {
                for (const v of sliceLevels[iLevel]) {
                    for (const u of dag.in_neighbors.get(v)) {
                        sliceEdges.push([u, v]);
                    }
                }
            }
            let result = newDagFromPrecedence(sliceLevels, sliceEdges);
            // Debug check:
            // for (const [u, v] of result.edges) {
            //     assert(result.vertices.includes(u), u, 'not in vertices');
            //     assert(result.vertices.includes(v), v, 'not in vertices');
            // }
            return result;
        }
        function getSlices(dag) {
            /* Find all n "slice-vertices" v[i] such that
                Vertex v is a slice-vertex <==> foreach vertex u. exists path u->v or exists path v-> or u==v
            Find all n+1 "slices" S[i] such that
                u in S[0] <==> exists path         u->v[0]
                u in S[i] <==> exists path v[i-1]->u->v[i]
                u in S[n] <==> exists path v[n-1]->u
            Return list S[0], v[0], S[1], v[1], ... v[n-1], S[n]. (ommitting empty S[i]) */
            let ancestors = new Map();
            let seenVertices = 0;
            let slices = [];
            let lastSliceVertex = -1;
            const nLevels = dag.levels.length;
            for (let iLevel = 0; iLevel < nLevels; iLevel++) {
                const level = dag.levels[iLevel];
                for (let iVertex = 0; iVertex < level.length; iVertex++) {
                    const v = level[iVertex];
                    const ancestors_v = new Set(dag.in_neighbors.get(v));
                    for (const u of dag.in_neighbors.get(v))
                        for (const t of ancestors.get(u))
                            ancestors_v.add(t);
                    ancestors.set(v, ancestors_v);
                }
                if (level.length == 1 && ancestors.get(level[0]).size == seenVertices) {
                    // This is a slice-vertex
                    const sliceVertex = level[0];
                    if (iLevel > lastSliceVertex + 1) {
                        slices.push(getSlice(dag, lastSliceVertex + 1, iLevel));
                    }
                    slices.push(newDagFromPrecedence([[sliceVertex]], []));
                    ancestors.clear();
                    ancestors.set(sliceVertex, new Set());
                    lastSliceVertex = iLevel;
                    seenVertices = 1;
                }
                else {
                    seenVertices += level.length;
                }
            }
            if (nLevels > lastSliceVertex + 1) {
                slices.push(getSlice(dag, lastSliceVertex + 1, nLevels));
            }
            // Debug check:
            // let checkVertices = new Set();
            // for (const slice of slices) for (const v of slice.vertices) checkVertices.add(v);
            // assert(setsEqual(checkVertices, new Set(dag.vertices)), 'checkVertices', checkVertices, '!= dag.vertices', dag.vertices);
            return slices;
        }
        function embedDag(dag, vertexSizes, padding = { x: 0.0, y: 0.0 }, leftMargin = 0.0, rightMargin = 0.0, topMargin = 0.0, bottomMargin = 0.0) {
            /* Place vertices in a plane so that all edges go from left to right, use heuristics to make it look nice.
            Return the bounding box and positions of individual vertices. */
            let box;
            let positions;
            if (dag.vertices.length == 0) {
                box = { left: 0.0, right: 0.0, top: 0.0, bottom: 0.0, weight: 0.0 };
                positions = new Map();
            }
            else {
                [box, positions] = embedGeneralDag(dag, vertexSizes, padding);
            }
            box.left += leftMargin;
            box.right += rightMargin;
            box.top += topMargin;
            box.bottom += bottomMargin;
            return [box, positions];
        }
        Graphs.embedDag = embedDag;
        function embedGeneralDag(dag, vertex_sizes, padding) {
            /* Embed non-empty DAG with possibly more than one component. */
            let components = dagConnectedComponents(dag);
            let embeddings = components.map(comp => embedConnectedDag(comp, vertex_sizes, padding));
            let grandEmbedding = combineEmbeddings(embeddings, Direction.VERTICAL_REARRANGED, padding);
            return grandEmbedding;
        }
        function embedConnectedDag(dag, vertexSizes, padding) {
            /* Embed non-empty DAG with exactly one component. */
            const slices = getSlices(dag);
            if (slices.length == 1) {
                return embedUnsliceableDag(slices[0], vertexSizes, padding);
            }
            else {
                const embeddings = slices.map(s => embedGeneralDag(s, vertexSizes, padding));
                const grandEmbedding = combineEmbeddings(embeddings, Direction.HORIZONTAL, padding);
                return grandEmbedding;
            }
        }
        function embedUnsliceableDag(dag, vertexSizes, padding, maxAllowedPermutations = 1024) {
            /* Embed non-empty DAG with exactly one component, without trying to slice it. */
            const nPerm = permutationNumber(dag.levels, maxAllowedPermutations + 1);
            if (dag.vertices.length == 1 || nPerm > maxAllowedPermutations) {
                // Do not permutate (not needed or too expensive)
                return embedLevels(dag.levels, vertexSizes, padding = padding);
            }
            else {
                // Permutate all levels to find the best tuple of permutations
                let bestEmbedding = null;
                let bestPenalty = Infinity;
                const levelPermutations = dag.levels.map(level => permutations(level, true));
                for (const perm of cartesianProduct(levelPermutations)) {
                    const embedding = embedLevels(perm, vertexSizes, padding);
                    const [box, positions] = embedding;
                    let penalty = 0.0;
                    for (const [u, v] of dag.edges)
                        penalty += Math.abs(positions.get(u).y - positions.get(v).y);
                    if (penalty < bestPenalty) {
                        bestEmbedding = embedding;
                        bestPenalty = penalty;
                    }
                }
                assert(bestEmbedding != null, 'bestEmbedding is null', bestEmbedding);
                return bestEmbedding;
            }
        }
        function embedLevels(levels, vertexSizes, padding) {
            const levelEmbeddings = [];
            for (const level of levels) {
                const vertexEmbeddings = [];
                for (const vertex of level) {
                    const { width, height, weight } = vertexSizes.get(vertex);
                    const box = { left: width / 2, right: width / 2, top: height / 2, bottom: height / 2, weight: weight };
                    const positions = new Map([[vertex, { x: 0.0, y: 0.0 }]]);
                    vertexEmbeddings.push([box, positions]);
                }
                const levelEmbedding = combineEmbeddings(vertexEmbeddings, Direction.VERTICAL, padding);
                levelEmbeddings.push(levelEmbedding);
            }
            const grandEmbedding = combineEmbeddings(levelEmbeddings, Direction.HORIZONTAL, padding);
            return grandEmbedding;
        }
        function permutationNumber(levels, maxAllowed) {
            /* Return the cardinality of the cartesian product of permutations of each level,
            or return maxAllowed if it becomes clear that the result would be > maxAllowed. */
            let result = 1;
            for (const level of levels) {
                for (let i = level.length; i > 0; i--) {
                    result *= i;
                    if (result > maxAllowed)
                        return maxAllowed;
                }
            }
            return result;
        }
        function permutations(elements, reversed = false) {
            if (elements.length == 0)
                return [[]];
            const result = [];
            let head = elements[0];
            let rest = elements.slice(1, undefined);
            for (let i = 0; i < elements.length; i++) {
                for (const subperm of permutations(rest, true)) {
                    subperm.push(head);
                    result.push(subperm);
                }
                [head, rest[i]] = [rest[i], head];
            }
            if (!reversed)
                for (const perm of result)
                    perm.reverse();
            return result;
        }
        function cartesianProduct(sets, skip = 0) {
            if (skip == sets.length)
                return [[]];
            const result = [];
            for (const head of sets[skip]) {
                for (const tuples of cartesianProduct(sets, skip + 1)) {
                    tuples.push(head);
                    result.push(tuples);
                }
            }
            if (skip == 0)
                for (const tup of result)
                    tup.reverse();
            return result;
        }
        let Direction;
        (function (Direction) {
            Direction[Direction["HORIZONTAL"] = 0] = "HORIZONTAL";
            Direction[Direction["VERTICAL"] = 1] = "VERTICAL";
            Direction[Direction["VERTICAL_REARRANGED"] = 2] = "VERTICAL_REARRANGED";
        })(Direction || (Direction = {}));
        function combineEmbeddings(embeddings, direction, padding) {
            const boxes = embeddings.map(emb => emb[0]);
            const positions = embeddings.map(emb => emb[1]);
            let grandBox;
            let boxPlaces;
            if (direction == Direction.HORIZONTAL)
                [grandBox, boxPlaces] = stackBoxesHorizontally(boxes, padding);
            else if (direction == Direction.VERTICAL)
                [grandBox, boxPlaces] = stackBoxesVertically(boxes, padding);
            else if (direction == Direction.VERTICAL_REARRANGED)
                [grandBox, boxPlaces] = stackBoxesVertically(boxes, padding, true);
            else
                throw new Error("direction must be HORIZONTAL or VERTICAL or VERTICAL_REARRANGED");
            let grandPositions = new Map();
            for (let i = 0; i < embeddings.length; i++) {
                for (const [vertex, xy] of positions[i]) {
                    grandPositions.set(vertex, addXY(boxPlaces[i], xy));
                }
            }
            return [grandBox, grandPositions];
        }
        function sum(array) {
            let result = 0;
            for (const x of array)
                result += x;
            return result;
        }
        function stackBoxesHorizontally(boxes, padding) {
            /* Return the grand box containing all boxes, and the list of positions of boxes in the grand box. */
            const grandWidth = sum(boxes.map(box => box.left + box.right)) + (boxes.length - 1) * padding.x;
            const grandLeft = grandWidth / 2;
            const grandRight = grandLeft;
            const grandTop = Math.max(...boxes.map(box => box.top));
            const grandBottom = Math.max(...boxes.map(box => box.bottom));
            const grandWeight = sum(boxes.map(box => box.weight));
            let boxPlaces = [];
            let x = -grandLeft;
            for (const box of boxes) {
                x += box.left;
                boxPlaces.push({ x: x, y: 0.0 });
                x += box.right;
                x += padding.x;
            }
            const grandBox = { left: grandLeft, right: grandRight, top: grandTop, bottom: grandBottom, weight: grandWeight };
            return [grandBox, boxPlaces];
        }
        function stackBoxesVertically(boxes, padding, rearrange = false) {
            /* Return the grand box containing all boxes, and the list of positions of boxes in the grand box. */
            const nBoxes = boxes.length;
            let sumW = 0.0;
            let sumWY = 0.0;
            let y = 0.0;
            const rearrangedBoxes = rearrange ? rearrangeBoxesFromMiddle(boxes) : boxes.map((box, i) => [i, box]);
            for (const [i, box] of rearrangedBoxes) {
                y += box.top;
                sumW += box.weight;
                sumWY += box.weight * y;
                y += box.bottom;
                y += padding.y;
            }
            const meanY = sumW > 0 ? sumWY / sumW : 0;
            const boxPlaces = [];
            y = -meanY;
            for (const [i, box] of rearrangedBoxes) {
                y += box.top;
                boxPlaces[i] = { x: 0.0, y: y };
                y += box.bottom;
                y += padding.y;
            }
            const grandTop = meanY;
            const grandBottom = y - padding.y;
            const grandLeft = Math.max(...boxes.map(box => box.left));
            const grandRight = Math.max(...boxes.map(box => box.right));
            const grandBox = { left: grandLeft, right: grandRight, top: grandTop, bottom: grandBottom, weight: sumW };
            return [grandBox, boxPlaces];
        }
        function rearrangeBoxesFromMiddle(boxes) {
            /* Change the order of the boxes so that the biggest are in the middle, e.g. 1 3 5 7 8 6 4 2.
            Return the list of tuples (original_index, box). */
            const indexedBoxes = boxes.map((box, i) => [i, box]);
            indexedBoxes.sort((a, b) => b[1].weight - a[1].weight); // sort from heaviest
            let even = [];
            let odd = [];
            for (let i = 0; i < indexedBoxes.length; i++) {
                if (i % 2 == 0)
                    even.push(indexedBoxes[i]);
                else
                    odd.push(indexedBoxes[i]);
            }
            odd.reverse();
            odd.push(...even);
            return odd;
        }
        function connectedComponents(vertices, edges) {
            /* Get connected components of a graph (efficient implementation with shallow tree).
            Connected components and the vertices within them are sorted. */
            const shallowTree = newShallowTree(vertices);
            for (const [u, v] of edges)
                stJoin(shallowTree, u, v);
            return stGetSets(shallowTree);
        }
        function dagConnectedComponents(dag) {
            const components = connectedComponents(dag.vertices, dag.edges);
            const nComponents = components.length;
            const vertex2component = new Map();
            for (let iComponent = 0; iComponent < components.length; iComponent++) {
                const component = components[iComponent];
                for (const v of component)
                    vertex2component.set(v, iComponent);
            }
            const dags = [];
            for (let i = 0; i < nComponents; i++)
                dags.push(newDagEmpty());
            for (let iLevel = 0; iLevel < dag.levels.length; iLevel++) {
                for (const vertex of dag.levels[iLevel]) {
                    const iComp = vertex2component.get(vertex);
                    const levs = dags[iComp].levels;
                    if (levs.length <= iLevel)
                        levs.push([]);
                    assert(levs.length == iLevel + 1, 'levs.length != iLevel + 1'); // Debug
                    levs[iLevel].push(vertex);
                    dags[iComp].vertices.push(vertex);
                    dags[iComp].in_neighbors.set(vertex, []);
                }
            }
            for (const edge of dag.edges) {
                const [u, v] = edge;
                const iComp = vertex2component.get(u);
                assert(vertex2component.get(v) == iComp, 'vertex2component.get(v) != iComp'); // Debug
                dags[iComp].edges.push(edge);
                dags[iComp].in_neighbors.get(v).push(u);
            }
            return dags;
        }
        /* Efficient representation of a set of disjoint sets.
        Each set is identified by its 'root', which is its smallest element.
        The Map maps each element to its root.*/
        function newShallowTree(elements) {
            let tree = new Map();
            for (const elem of elements)
                tree.set(elem, elem);
            return tree;
        }
        function stRoot(self, element) {
            const parent = self.get(element);
            if (parent == element) { // This element is the root
                return element;
            }
            else {
                const root = stRoot(self, parent);
                self.set(element, root);
                return root;
            }
        }
        function stJoin(self, i, j) {
            /* Join the set containing i and the set containing j, return the root of the new set. (Do nothing if i and j already are in the same set.) */
            const rootI = stRoot(self, i);
            const rootJ = stRoot(self, j);
            const newRoot = Math.min(rootI, rootJ);
            self.set(rootI, newRoot);
            self.set(rootJ, newRoot);
            return newRoot;
        }
        function stGetSets(self) {
            /* Return the list of sets, each set itself is a list of elements. */
            const setMap = new Map();
            for (const elem of self.keys()) {
                const root = stRoot(self, elem);
                if (setMap.has(root))
                    setMap.get(root).push(elem);
                else
                    setMap.set(root, [elem]);
            }
            const setList = [...setMap.values()];
            for (const set of setList)
                set.sort();
            setList.sort();
            return setList;
        }
        function assert(assertion, ...details) {
            if (!assertion) {
                console.error('AssertionError:', ...details);
                throw new Error('AssertionError');
            }
        }
    })(Graphs || (Graphs = {}));

    var __awaiter = (undefined && undefined.__awaiter) || function (thisArg, _arguments, P, generator) {
        function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    };
    var Drawing;
    (function (Drawing) {
        function zoomOut(viewer, ratio = Constants.ZOOM_STEP_RATIO, mouseXY) {
            Geometry.zoomInfoZoomOut(viewer.zoom, ratio);
            let [relPivotX, relPivotY] = mouseXY != undefined ?
                [mouseXY[0] / viewer.screen.width, mouseXY[1] / viewer.screen.height]
                : [0.5, 0.5];
            let newWidth = viewer.screen.width * viewer.zoom.xZoomout;
            let newHeight = viewer.screen.height * viewer.zoom.yZoomout;
            let { x, y, width, height } = viewer.visWorld;
            viewer.visWorld.x = x + (width - newWidth) * relPivotX;
            viewer.visWorld.y = y + (height - newHeight) * relPivotY;
            viewer.visWorld.width = newWidth;
            viewer.visWorld.height = newHeight;
            constrainVisWorldXY(viewer);
            redraw(viewer);
            updateZoomButtons(viewer);
        }
        Drawing.zoomOut = zoomOut;
        function zoomIn(viewer, ratio = Constants.ZOOM_STEP_RATIO, mouseXY) {
            zoomOut(viewer, 1 / ratio, mouseXY);
        }
        Drawing.zoomIn = zoomIn;
        function zoomAll(viewer) {
            Geometry.zoomInfoSetZoomout(viewer.zoom, viewer.zoom.initialZoomout);
            viewer.visWorld = Geometry.centeredVisWorld(viewer.world, viewer.screen, viewer.zoom);
            redraw(viewer);
            updateZoomButtons(viewer);
        }
        Drawing.zoomAll = zoomAll;
        function zoomSet(viewer, newZoomout, centerXY) {
            Geometry.zoomInfoSetZoomout(viewer.zoom, newZoomout);
            let [centerX, centerY] = centerXY !== null && centerXY !== void 0 ? centerXY : Geometry.rectangleCenter(viewer.visWorld);
            let visWidth = viewer.screen.width * viewer.zoom.xZoomout;
            let visHeight = viewer.screen.height * viewer.zoom.yZoomout;
            viewer.visWorld = {
                x: centerX - 0.5 * visWidth,
                y: centerY - 0.5 * visHeight,
                width: visWidth,
                height: visHeight,
            };
            constrainVisWorldXY(viewer);
            redraw(viewer, false);
            updateZoomButtons(viewer);
        }
        Drawing.zoomSet = zoomSet;
        function updateZoomButtons(viewer) {
            viewer.mainDiv.selectAll('div.button#zoom-out').attr('disabled', viewer.zoom.zoomout >= viewer.zoom.maxZoomout ? 'true' : null);
            viewer.mainDiv.selectAll('div.button#zoom-in').attr('disabled', viewer.zoom.zoomout <= viewer.zoom.minZoomout ? 'true' : null);
            viewer.mainDiv.selectAll('div.button#zoom-reset').attr('disabled', viewer.zoom.zoomout == viewer.zoom.initialZoomout ? 'true' : null);
        }
        function shift(viewer, rightRelative, downRelative) {
            let { width, height } = viewer.visWorld;
            viewer.visWorld.x += rightRelative * width;
            viewer.visWorld.y += downRelative * height;
            constrainVisWorldXY(viewer);
            redraw(viewer);
        }
        Drawing.shift = shift;
        function shiftByMouse(viewer, rightPixels, downPixels) {
            let { width, height } = viewer.visWorld;
            let { width: sw, height: sh } = viewer.screen;
            viewer.visWorld.x += rightPixels * width / sw;
            viewer.visWorld.y += downPixels * height / sh;
            constrainVisWorldXY(viewer);
            redraw(viewer, false);
        }
        Drawing.shiftByMouse = shiftByMouse;
        function save(viewer) {
            const w = viewer.screen.width;
            const h = viewer.screen.height;
            const serializer = new XMLSerializer();
            viewer.canvas.attr('rendering', 'rendering');
            const svgString = serializer.serializeToString(viewer.canvas.node());
            viewer.canvas.attr('rendering', null);
            const img = new Image(w, h);
            img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgString)));
            // const canvas = viewer.mainDiv.append('canvas').attr('width', w).attr('height', h).styles({position: 'absolute', left: '0', top: '0', 'z-index': '999'});
            const canvas = d3.select(document.createElement('canvas')).attr('width', w).attr('height', h); // not attaching canvas to DOM
            d3.select(img).on('load', () => {
                canvas.node().getContext('2d').drawImage(img, 0, 0, w, h);
                const imgData = canvas.node().toDataURL("image/png").replace("image/png", "image/octet-stream");
                saveFile(imgData, 'overprot.png');
                // d3.select(window).on('focus.removecanvas', () => {
                //     console.log('picovina'); 
                //     canvas.remove();
                //     d3.select(window).on('focus.removecanvas', null);
                // });
            });
        }
        Drawing.save = save;
        function saveFile(data, fileName) {
            const saveLink = document.createElement("a");
            saveLink.download = fileName;
            saveLink.href = data;
            saveLink.click();
        }
        function placeTooltip(viewer, tooltip) {
            tooltip = tooltip || viewer.mainDiv.select('div.tooltip');
            return tooltip
                .style('left', (d3.event.clientX + Constants.TOOLTIP_OFFSET.x))
                .style('top', (d3.event.clientY + Constants.TOOLTIP_OFFSET.y));
        }
        function tooltipMouseEnter(viewer, targetElement, htmlContent, delay = false) {
            var _a;
            let hasPinnedTooltip = d3.select(targetElement).attr('tooltip') == 'pinned';
            if (!hasPinnedTooltip) {
                let tooltip = viewer.mainDiv.append('div').attr('class', 'tooltip').attr('type', 'hover').html(htmlContent);
                placeTooltip(viewer, tooltip);
                (_a = d3.select(targetElement)) === null || _a === void 0 ? void 0 : _a.attr('tooltip', 'hover');
                fadeIn(tooltip, delay ? Constants.TOOLTIP_DELAY : 0);
            }
        }
        function tooltipMouseMove(viewer) {
            let tooltip = viewer.mainDiv.selectAll('div.tooltip[type=hover]');
            placeTooltip(viewer, tooltip);
        }
        function tooltipMouseLeave(viewer) {
            viewer.mainDiv.selectAll('div.tooltip[type=hover]').remove();
            viewer.mainDiv.selectAll('[tooltip=hover]').attr('tooltip', null);
        }
        function tooltipMouseClick(viewer, targetElement, htmlContent) {
            var _a;
            // add pre-pinned tooltip
            let tooltip = viewer.mainDiv.append('div').attr('class', 'tooltip').attr('type', 'pre-pinned').html(htmlContent);
            placeTooltip(viewer, tooltip);
            (_a = d3.select(targetElement)) === null || _a === void 0 ? void 0 : _a.attr('tooltip', 'pre-pinned');
            // add listener to GUI to remove all pinned tooltips on click and remove all tooltip='pinned' attributes and change pre-pinned to pinned - will be invoked in this bubbling
            viewer.mainDiv.on('click.tooltip', () => {
                viewer.mainDiv.selectAll('div.tooltip[type=hover],div.tooltip[type=pinned]').remove();
                viewer.mainDiv.selectAll('div.tooltip[type=pre-pinned]').attr('type', 'pinned');
                viewer.mainDiv.selectAll('[tooltip=hover],[tooltip=pinned]').attr('tooltip', null);
                viewer.mainDiv.selectAll('[tooltip=pre-pinned]').attr('tooltip', 'pinned');
            });
        }
        function unpinAllTooltips(viewer) {
            viewer.mainDiv.selectAll('div.tooltip[type=pinned]').remove();
            viewer.mainDiv.selectAll('[tooltip=pinned]').attr('tooltip', null);
        }
        function setTooltips(viewer, selection, htmlContents, pinnable = false, delay = false) {
            if (htmlContents === null) {
                selection
                    .on('mouseenter.tooltip', null)
                    .on('mousemove.tooltip', null)
                    .on('mouseleave.tooltip', null)
                    .on('click.tooltip', null);
            }
            else {
                let active = htmlContents.map(content => content != null);
                let activeContents = htmlContents.filter(content => content != null);
                selection.filter((d, i) => active[i])
                    .on('mouseenter.tooltip', (d, i, g) => tooltipMouseEnter(viewer, g[i], activeContents[i], delay))
                    .on('mousemove.tooltip', (d, i) => tooltipMouseMove(viewer))
                    .on('mouseleave.tooltip', (d, i) => tooltipMouseLeave(viewer));
                if (pinnable) {
                    selection.filter((d, i) => active[i])
                        .on('click.tooltip', (d, i, g) => tooltipMouseClick(viewer, g[i], activeContents[i]));
                }
                selection.filter((d, i) => !active[i])
                    .on('mouseenter.tooltip', null)
                    .on('mousemove.tooltip', null)
                    .on('mouseleave.tooltip', null)
                    .on('click.tooltip', null);
            }
        }
        Drawing.setTooltips = setTooltips;
        function addPointBehavior(viewer, selection, pointedElementSelector = (pointed) => d3.select(pointed), callback = null) {
            selection.on('mouseenter.point', (d, i, g) => {
                let pointed = pointedElementSelector(g[i]);
                pointed.attr('pointed', 'pointed');
                if (callback != null)
                    callback(pointed);
            });
            selection.on('mouseleave.point', (d, i, g) => {
                pointedElementSelector(g[i]).attr('pointed', null);
                if (callback != null)
                    callback(d3.selectAll());
            });
        }
        Drawing.addPointBehavior = addPointBehavior;
        function addPickBehavior(viewer, selection, pickedElementSelector = (clicked) => d3.select(clicked), callback = null) {
            selection.on('click.pick', (d, i, g) => {
                let d3Elem = pickedElementSelector(g[i]);
                d3Elem.attr('picked', 'pre-picked');
            });
            viewer.guiDiv
                .on('click.pick', () => {
                viewer.guiDiv.selectAll('[picked=picked]').attr('picked', null);
                let picked = viewer.guiDiv.selectAll('[picked=pre-picked]');
                picked.attr('picked', 'picked');
                if (callback != null)
                    callback(picked);
            });
        }
        Drawing.addPickBehavior = addPickBehavior;
        function addMouseHoldBehavior(selection, onDown, onHold, onUp) {
            selection.on('mousedown', () => __awaiter(this, void 0, void 0, function* () {
                if (d3.event.which == 1 || d3.event.which == undefined) { // d3.event.which: 1=left, 2=middle, 3=right mouse button
                    let thisClickId = Math.random().toString(36).slice(2);
                    onDown();
                    selection.attr('pressed', thisClickId);
                    yield sleep(Constants.MOUSE_HOLD_BEHAVIOR_INITIAL_SLEEP_TIME);
                    while (selection.attr('pressed') == thisClickId) {
                        onHold();
                        yield sleep(Constants.MOUSE_HOLD_BEHAVIOR_STEP_SLEEP_TIME);
                    }
                }
            }));
            selection.on('mouseup', () => __awaiter(this, void 0, void 0, function* () {
                selection.attr('pressed', null);
                onUp();
            }));
            selection.on('mouseleave', () => __awaiter(this, void 0, void 0, function* () {
                selection.attr('pressed', null);
                onUp();
            }));
        }
        Drawing.addMouseHoldBehavior = addMouseHoldBehavior;
        function sleep(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        }
        function recolor(viewer, transition = true) {
            let duration = transition ? Constants.TRANSITION_DURATION : 0;
            viewer.canvas
                .select('g.nodes')
                .selectAll('g.node')
                .selectAll('rect,polygon')
                .transition().duration(duration)
                .style('fill', n => n.visual.fill)
                .style('stroke', n => n.visual.stroke);
            let betaArcs = viewer.canvas
                .select('g.beta-connectivity')
                .selectAll('g.ladder')
                .selectAll('path.vis');
            // if (viewer.settings.colorMethod == Enums.ColorMethod.Stdev || viewer.settings.colorMethod == Enums.ColorMethod.Rainbow) {
            //     betaArcs.style('stroke', Colors.NEUTRAL_DARK.hex());
            // } else {
            //     betaArcs.style('stroke', ladder => viewer.data.nodes[(ladder as Dag.Edge)[0]].visual.stroke);
            // }
            betaArcs.style('stroke', ladder => arcColor(viewer, ladder));
            show3DVariabilityLegend(viewer, viewer.settings.colorMethod == Enums.ColorMethod.Stdev, transition);
        }
        Drawing.recolor = recolor;
        function arcColor(viewer, ladder) {
            if (viewer.settings.colorMethod == Enums.ColorMethod.Stdev || viewer.settings.colorMethod == Enums.ColorMethod.Rainbow) {
                return Colors.NEUTRAL_DARK.hex();
            }
            else {
                return viewer.data.nodes[ladder[0]].visual.stroke;
            }
        }
        function redraw(viewer, transition = true) {
            let duration = transition ? Constants.TRANSITION_DURATION : 0;
            let d3nodes = viewer.canvas
                .select('g.nodes')
                .selectAll('g.node')
                .filter(n => n.active);
            d3nodes.select('rect')
                .transition().duration(duration)
                .attrs(n => Geometry.rectToScreen(viewer.visWorld, viewer.screen, n.visual.rect));
            d3nodes.select('polygon')
                .transition().duration(duration)
                .attr('points', n => Geometry.symCdfPolygonPoints(Geometry.rectToScreen(viewer.visWorld, viewer.screen, n.visual.rect), n.cdf));
            let labelVisibility = d3nodes.select('text').data().map(n => nodeBigEnoughForLabel(viewer, n));
            d3nodes.select('text')
                .style('fill', Constants.NODE_LABEL_COLOR) // fill must be set both before and after opacity transition because fill doesn't support transitions
                .transition().duration(duration)
                .attrs(n => {
                let { x, y, height, width } = Geometry.rectToScreen(viewer.visWorld, viewer.screen, n.visual.rect);
                return { x: x + width / 2, y: y + height + Constants.HANGING_TEXT_OFFSET };
            })
                .style('opacity', (n, i) => labelVisibility[i] ? 1 : 0)
                .transition().duration(0)
                .style('fill', (n, i) => labelVisibility[i] ? Constants.NODE_LABEL_COLOR : 'none');
            viewer.canvas
                .select('g.edges')
                .selectAll('line')
                .transition().duration(duration)
                .attrs(line => Geometry.lineToScreen(viewer.visWorld, viewer.screen, line));
            let arcMaxMinor = Constants.ARC_MAX_MINOR / viewer.zoom.yZoomout;
            // let arcMaxDeviation = Constants.ARC_MAX_DEVIATION / viewer.zoom.yZoomout;
            // let arcSmartDeviationParam = viewer.world.width * Constants.ARC_SMART_DEVIATION_PARAM_WRT_WORLD_WIDTH / viewer.zoom.xZoomout;
            let dag = viewer.data;
            dag.beta_connectivity.forEach(edge => edge[3] = dag.nodes[edge[0]].active && dag.nodes[edge[1]].active ? 1 : 0);
            viewer.canvas
                .select('g.beta-connectivity')
                .selectAll('path')
                .transition().duration(duration)
                .attr('d', edge => calculateArcPath(viewer, edge));
        }
        Drawing.redraw = redraw;
        function calculateArcPath(viewer, edge) {
            if (edge[3] == 0)
                return '';
            let [u, v, orientation] = edge;
            let n1 = viewer.data.nodes[u].visual.rect;
            let n2 = viewer.data.nodes[v].visual.rect;
            let endpoints = Geometry.lineToScreen(viewer.visWorld, viewer.screen, { x1: n1.x + 0.5 * n1.width, y1: n1.y + 0.5 * n1.height, x2: n2.x + 0.5 * n2.width, y2: n2.y + 0.5 * n2.height });
            let x2yZoomRatio = viewer.zoom.yZoomout / viewer.zoom.xZoomout;
            let arcMaxMinor = Constants.ARC_MAX_MINOR / viewer.zoom.yZoomout;
            // return Geometry.arcPathD_circle(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
            return Geometry.arcPathD_ellipse(endpoints, viewer.world.width / viewer.zoom.xZoomout, arcMaxMinor, orientation == 1, x2yZoomRatio);
            // return Geometry.arcPathD_bezier(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
        }
        function nodeBigEnoughForLabel(viewer, node) {
            return Geometry.rectToScreen(viewer.visWorld, viewer.screen, node.visual.rect).width >= Constants.MINIMAL_WIDTH_FOR_SSE_LABEL;
        }
        Drawing.nodeBigEnoughForLabel = nodeBigEnoughForLabel;
        function gradientBarExp(canvas, rect, maxValue, middle, ticksDistance, legend = '') {
            let bar = canvas.append('g').attr('class', 'heatmap-bar')
                .attr('transform', `translate(${rect.x},${rect.y})`);
            let barLabel = bar.append('text').attr('class', 'heatmap-bar-label')
                .attrs({ x: -5, y: 0.5 * rect.height })
                .text(legend);
            let n = Math.floor(rect.width);
            let step = (rect.width - 1) / n;
            let barFill = bar.append('g').attr('class', 'heatmap-bar-fill');
            let barStroke = bar.append('g').attr('class', 'heatmap-bar-stroke');
            barFill.selectAll('rect')
                .data(d3.range(n))
                .enter()
                .append('rect')
                .attrs(i => { return { x: i * step, y: 0, width: step + 1, height: rect.height, stroke: 'none', fill: Colors.byExpHeatmap(i / n * maxValue, middle).hex() }; });
            barStroke.append('rect')
                .attrs(i => { return { x: 0, y: 0, width: rect.width, height: rect.height, fill: 'none' }; });
            let barTicks = bar.append('g').attr('class', 'heatmap-bar-ticks');
            let barTickTexts = barTicks.selectAll('text')
                .data(d3.range(Math.floor(maxValue / ticksDistance) + 1))
                .enter()
                .append('text')
                .attr('x', i => i * ticksDistance / maxValue * rect.width)
                .attr('y', rect.height + Constants.HANGING_TEXT_OFFSET)
                .text(i => i * ticksDistance);
            return bar;
        }
        function moveGradientBarExp(bar, newX, newY, ticksAbove) {
            bar.attr('transform', `translate(${newX},${newY})`);
            let barTickTexts = bar.select('g.heatmap-bar-ticks').selectAll('text');
            let barHeight = bar.select('g.heatmap-bar-stroke rect').attr('height');
            if (ticksAbove) {
                barTickTexts.attr('y', -Constants.HANGING_TEXT_OFFSET);
                barTickTexts.style('dominant-baseline', 'alphabetic'); // For some reason 'alphabetic' means 'bottom'
            }
            else {
                barTickTexts.attr('y', barHeight + Constants.HANGING_TEXT_OFFSET);
                barTickTexts.style('dominant-baseline', 'hanging');
            }
        }
        function show3DVariabilityLegend(viewer, on, transition = true) {
            var _a, _b, _c, _d;
            fadeOutRemove(viewer.canvas.selectAll('g.heatmap-bar'));
            let BAR_WIDTH = 200;
            let BAR_HEIGHT = 20;
            let BAR_HMARGIN = 15;
            let BAR_VMARGIN = 5;
            if (on) {
                let bar = gradientBarExp(viewer.canvas, { x: viewer.screen.width - BAR_WIDTH - BAR_HMARGIN, y: BAR_VMARGIN, width: BAR_WIDTH, height: BAR_HEIGHT }, 15, 5, 5, '3D variability [\u212B]');
                let controlsRight = (_b = (_a = viewer.mainDiv.select('div.control-panel#main-panel').node()) === null || _a === void 0 ? void 0 : _a.getBoundingClientRect()) === null || _b === void 0 ? void 0 : _b.right;
                let barLeft = (_d = (_c = bar.node()) === null || _c === void 0 ? void 0 : _c.getBoundingClientRect()) === null || _d === void 0 ? void 0 : _d.left;
                if (controlsRight !== undefined && barLeft !== undefined && controlsRight > barLeft) {
                    moveGradientBarExp(bar, viewer.screen.width - BAR_WIDTH - BAR_HMARGIN, viewer.screen.height - BAR_VMARGIN - BAR_HEIGHT, true);
                }
                if (transition) {
                    fadeIn(bar);
                }
            }
        }
        function showBetaConnectivity(viewer, on, transition = true) {
            viewer.settings.betaConnectivityVisibility = on;
            let oldBetaConnectivityVis = viewer.canvas.selectAll('g.beta-connectivity');
            if (oldBetaConnectivityVis.size() > 0 && !on) {
                if (transition)
                    fadeOutRemove(oldBetaConnectivityVis);
                else
                    oldBetaConnectivityVis.remove();
            }
            else if (oldBetaConnectivityVis.size() == 0 && on) {
                let dag = viewer.data;
                dag.beta_connectivity.forEach(edge => edge[3] = dag.nodes[edge[0]].active && dag.nodes[edge[1]].active ? 1 : 0);
                let betaConnectivityVis = viewer.canvas
                    .append('g').attr('class', 'beta-connectivity');
                let betaConnectivityLadders = betaConnectivityVis.selectAll()
                    .data(dag.beta_connectivity)
                    .enter()
                    .append('g').attr('class', 'ladder');
                viewer.ladderMap = new Types.TupleMap();
                betaConnectivityLadders.each((d, i, elems) => {
                    const strand1 = viewer.data.nodes[d[0]].label;
                    const strand2 = viewer.data.nodes[d[1]].label;
                    const dir = d[2] > 0 ? 'parallel' : 'antiparallel';
                    viewer.ladderMap.set([strand1, strand2, dir], elems[i]);
                });
                // console.log('ladderMap', viewer.ladderMap.entries());
                let betaPaths = betaConnectivityLadders
                    .append('path').attr('class', 'vis')
                    .style('stroke', ladder => arcColor(viewer, ladder));
                let betaGhostPaths = betaConnectivityLadders
                    .append('path').attr('class', 'ghost');
                // addPointBehavior(viewer, betaConnectivityPaths as any);
                addPointBehavior(viewer, betaGhostPaths, path => selectLadderFromGhostPath(viewer, path, true), nodes => Drawing.dispatchMixedEvent(viewer, Constants.EVENT_TYPE_HOVER, nodes.data()));
                redraw(viewer, false);
                if (transition) {
                    fadeIn(betaConnectivityVis);
                }
            }
        }
        Drawing.showBetaConnectivity = showBetaConnectivity;
        function selectLadderFromGhostPath(viewer, path, includeStrands) {
            let ladder = d3.select(path.parentElement);
            if (!includeStrands) {
                return ladder;
            }
            const edge = ladder.data()[0];
            let nodes = viewer.canvas.selectAll('g.node').nodes();
            const m = d3.selectAll([ladder.node(), nodes[edge[0]], nodes[edge[1]]]);
            return m;
        }
        function selectNodeFromShape(viewer, shape, includeLadders) {
            let gnode = d3.select(shape.parentElement);
            if (!includeLadders) {
                return gnode;
            }
            let result = [gnode.node()];
            const node = gnode.data()[0];
            if (node.ladders != undefined && node.ladders.length > 0) {
                let gladders = viewer.canvas.selectAll('g.ladder').nodes();
                for (const i of node.ladders)
                    result.push(gladders[i]);
            }
            return d3.selectAll(result);
        }
        Drawing.selectNodeFromShape = selectNodeFromShape;
        function fadeOutRemove(selection, delay = 0) {
            return selection.transition().delay(delay).duration(Constants.TRANSITION_DURATION).style('opacity', 0).remove();
        }
        Drawing.fadeOutRemove = fadeOutRemove;
        function fadeIn(selection, delay = 0) {
            if (selection.size() == 0) {
                return selection.transition();
            }
            let op = selection.style('opacity');
            return selection
                .style('opacity', 0)
                .transition().delay(delay).duration(Constants.TRANSITION_DURATION)
                .style('opacity', op);
        }
        Drawing.fadeIn = fadeIn;
        function constrainVisWorldXY(viewer) {
            let { x: wx, y: wy, width: ww, height: wh } = viewer.world;
            let { x: vx, y: vy, width: vw, height: vh } = viewer.visWorld;
            if (vw < ww * Constants.MAX_X_ZOOMOUT) {
                viewer.visWorld.x = Geometry.constrain(vx, wx - Constants.MAX_EMPTY_X_MARGIN * vw, wx + ww + Constants.MAX_EMPTY_X_MARGIN * vw - vw);
            }
            else {
                viewer.visWorld.x = wx + 0.5 * ww - 0.5 * vw;
            }
            if (vh < wh * Constants.MAX_Y_ZOOMOUT) {
                viewer.visWorld.y = Geometry.constrain(vy, wy - Constants.MAX_EMPTY_Y_MARGIN * vh, wy + wh + Constants.MAX_EMPTY_Y_MARGIN * vh - vh);
            }
            else {
                viewer.visWorld.y = wy + 0.5 * wh - 0.5 * vh;
            }
        }
        function dispatchMixedEvent(viewer, eventType, targets) {
            var _a;
            if (!viewer.settings.dispatchEvents)
                return;
            let sses = [];
            let edges = [];
            for (const target of targets) {
                if (target.label != undefined) { // is SSE
                    const sse = target;
                    sses.push({
                        label: sse.label,
                        type: sse.type,
                        sheetId: sse.type.toLowerCase() == 'e' ? sse.sheet_id : null,
                    });
                }
                else { // is ladder
                    const edge = target;
                    edges.push([
                        viewer.data.nodes[edge[0]].label,
                        viewer.data.nodes[edge[1]].label,
                        edge[2] > 0 ? 'parallel' : 'antiparallel',
                    ]);
                }
            }
            let eventDetail = {
                sourceType: (_a = viewer.d3viewer.node()) === null || _a === void 0 ? void 0 : _a.tagName.toLowerCase(),
                sourceId: viewer.id,
                sourceInternalId: viewer.internalId,
                eventType: eventType,
                sses: sses,
                ladders: edges,
            };
            viewer.d3viewer.dispatch(Constants.EVENT_PREFIX + eventType, { detail: eventDetail, bubbles: true });
        }
        Drawing.dispatchMixedEvent = dispatchMixedEvent;
        function handleEvent(viewer, event) {
            var _a, _b, _c, _d;
            // console.log('Inbound event', event.type, event);
            const detail = event.detail;
            if (detail == null || detail == undefined) {
                console.error(`Event ${event.type}: event.detail must be an object.`);
                return;
            }
            if (detail.sourceType == ((_a = viewer.d3viewer.node()) === null || _a === void 0 ? void 0 : _a.tagName.toLowerCase()) && detail.sourceInternalId == viewer.internalId) {
                // console.log('Ignoring self', viewer.uniqueId);
                return;
            }
            const sses = (_b = detail.sses) !== null && _b !== void 0 ? _b : [];
            const ladders = (_c = detail.ladders) !== null && _c !== void 0 ? _c : [];
            const PDB_OVERPROT_DO_SELECT = Constants.EVENT_PREFIX + Constants.EVENT_TYPE_DO_SELECT;
            const PDB_OVERPROT_DO_HOVER = Constants.EVENT_PREFIX + Constants.EVENT_TYPE_DO_HOVER;
            let attribute;
            switch (event.type) {
                case PDB_OVERPROT_DO_SELECT: // PDB.overprot.do.select
                    attribute = 'picked';
                    break;
                case PDB_OVERPROT_DO_HOVER: // PDB.overprot.do.hover
                    attribute = 'pointed';
                    break;
                default:
                    console.error('Unknown event type for OverProtViewer:', event.type);
                    return;
            }
            viewer.canvas.selectAll(`g.node[${attribute}],g.ladder[${attribute}]`).attr(attribute, null);
            for (const sse of sses) {
                if (sse.label == undefined) {
                    console.error(`Event ${event.type}: event.detail.sses[i].label must be a string.`);
                    return;
                }
                const g = viewer.nodeMap.get(sse.label);
                if (g != undefined) {
                    d3.select(g).attr(attribute, attribute);
                }
                else {
                    console.warn(`Event ${event.type}: SSE with label "${sse.label}" is not present.`);
                }
            }
            for (const ladder of ladders) {
                if (ladder.length < 3) {
                    console.error(`Event ${event.type}: event.detail.ladders[i] must be an array of length 3 ([strand1, strand2, "parallel"|"antiparallel"]).`, ladder);
                    return;
                }
                const u = ladder[0];
                const v = ladder[1];
                const dir = ladder[2];
                const g = (_d = viewer.ladderMap.get([u, v, dir])) !== null && _d !== void 0 ? _d : viewer.ladderMap.get([v, u, dir]);
                if (g != undefined) {
                    d3.select(g).attr(attribute, attribute);
                }
                else {
                    console.warn(`Event ${event.type}: Beta-ladder ${ladder} is not present.`);
                }
            }
            if (event.type == PDB_OVERPROT_DO_SELECT) {
                unpinAllTooltips(viewer);
            }
        }
        Drawing.handleEvent = handleEvent;
    })(Drawing || (Drawing = {}));

    var Controls;
    (function (Controls) {
        function emptySelection() {
            return d3.selectAll();
        }
        function newControlPanel(viewer, id, tooltip) {
            let panel = {
                base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showControlPanel(panel, par) }
            };
            return panel;
        }
        Controls.newControlPanel = newControlPanel;
        function addToControlPanel(panel, child) {
            panel.base.children.push(child);
        }
        Controls.addToControlPanel = addToControlPanel;
        function showControlPanel(panel, parentDiv) {
            panel.base.div = parentDiv.append('div').attr('class', 'control-panel').attr('id', panel.base.id);
            panel.base.children.forEach(child => child.base.show(panel.base.div));
        }
        function newButton(viewer, id, text, square, icon, onClick, tooltip) {
            let button = {
                base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showButton(button, par) },
                text: text,
                square: square,
                icon: icon,
                onClick: onClick
            };
            return button;
        }
        Controls.newButton = newButton;
        function showButton(button, parentDiv) {
            button.base.div = parentDiv.append('div').attr('class', button.square ? 'button square-button' : 'button').attr('id', button.base.id);
            const clas = button.icon ? 'button-icon' : 'button-text';
            button.base.div.append('div').attr('class', clas).html(button.text);
            Drawing.setTooltips(button.base.viewer, button.base.div, [button.base.tooltip], false, true);
            button.base.div.on('click', button.onClick);
            button.base.div.on('dblclick', () => { d3.event.stopPropagation(); });
        }
        function changeButtonText(button, newText) {
            button.text = newText;
            button.base.div.select('div.button-icon,div.button-text').html(newText);
        }
        Controls.changeButtonText = changeButtonText;
        function newPopup(viewer, id, text, autocollapse, tooltip) {
            // console.log('newPopup');
            let popup = {
                base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showPopup(popup, par) },
                headButton: newButton(viewer, null, text + Constants.OPEN_POPUP_SYMBOL, false, false, () => togglePopup(popup), tooltip),
                autocollapse: autocollapse
            };
            return popup;
        }
        Controls.newPopup = newPopup;
        function addToPopup(popup, child) {
            // console.log('addToPopup', popup, child);
            popup.base.children.push(child);
        }
        Controls.addToPopup = addToPopup;
        function showPopup(popup, parentDiv) {
            popup.base.div = parentDiv.append('div').attr('class', popup.autocollapse ? 'popup autocollapse' : 'popup').attr('id', popup.base.id);
            let headDiv = popup.base.div.append('div').attr('class', 'popup-head');
            popup.base.div.data([popup]);
            showButton(popup.headButton, headDiv);
        }
        function togglePopup(popup) {
            // console.log('togglePopup', popup);
            let expanded = popup.base.div.selectAll('div.popup-tail').size() > 0;
            if (expanded) {
                collapsePopup(popup);
            }
            else {
                collapseAllAutocollapsePopups(popup.base.viewer);
                expandPopup(popup);
                d3.event.stopPropagation();
            }
        }
        function expandPopup(popup) {
            // console.log('expandPopup');
            let headDiv = popup.base.div.select('div.popup-head');
            let tailDiv = popup.base.div.append('div').attr('class', 'popup-tail');
            popup.base.children.forEach(child => child.base.show(tailDiv));
            // let headWidth = headDiv.node().getBoundingClientRect().width;
            // let tailWidth = tailDiv.node()?.getBoundingClientRect().width;
            // if (tailWidth !== undefined && headWidth !== undefined && tailWidth < headWidth) {
            //     tailDiv.style('width', headWidth);
            // }
            if (popup.autocollapse) {
                popup.base.viewer.mainDiv.on('click.autocollapse-popups', () => collapseAllAutocollapsePopups(popup.base.viewer));
                tailDiv.on('click.autocollapse-popups', () => d3.event.stopPropagation());
            }
        }
        function collapsePopup(popup) {
            // console.log('collapsePopup');
            popup.base.div.selectAll('div.popup-tail').remove();
        }
        function collapseAllAutocollapsePopups(viewer) {
            viewer.mainDiv.selectAll('div.popup.autocollapse').each(d => collapsePopup(d));
        }
        function changePopupText(popup, newText) {
            changeButtonText(popup.headButton, newText + Constants.OPEN_POPUP_SYMBOL);
        }
        function newListbox(viewer, id, namesValuesTooltips, selectedValue, onSelect, tooltip) {
            let listbox = {
                base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showListbox(listbox, par) },
                namesValuesTooltips: namesValuesTooltips,
                selectedValue: selectedValue,
                onSelect: onSelect
            };
            return listbox;
        }
        Controls.newListbox = newListbox;
        function showListbox(listbox, parentDiv) {
            listbox.base.div = parentDiv.append('div').attr('class', 'listbox').attr('id', listbox.base.id);
            let itemOnClick = (nvt) => {
                listbox.selectedValue = nvt[1];
                listbox.base.div.selectAll('div.listbox-item')
                    .attr('class', nvt2 => nvt2[1] == listbox.selectedValue ? 'button listbox-item selected' : 'button listbox-item');
                listbox.onSelect(nvt[1]);
            };
            listbox.base.div.selectAll().data(listbox.namesValuesTooltips)
                .enter().append('div').attr('class', kv => kv[1] == listbox.selectedValue ? 'button listbox-item selected' : 'button listbox-item')
                .on('click', itemOnClick)
                .append('div').attr('class', 'button-text').html(kv => kv[0]);
            Drawing.setTooltips(listbox.base.viewer, listbox.base.div.selectAll('div.listbox-item'), listbox.namesValuesTooltips.map(nvt => nvt[2]), false, true);
            Drawing.setTooltips(listbox.base.viewer, listbox.base.div, [listbox.base.tooltip], false, true);
        }
        function newDropdownList(viewer, id, text, namesValuesTooltips, selectedValue, onSelect, tooltip, autocollapse = true, collapseOnSelect = true) {
            let popup = newPopup(viewer, id, text, autocollapse, tooltip);
            let wrappedOnSelect = collapseOnSelect ?
                (value) => { collapsePopup(popup); onSelect(value); }
                : onSelect;
            let listbox = newListbox(viewer, id, namesValuesTooltips, selectedValue, wrappedOnSelect, null);
            addToPopup(popup, listbox);
            return popup;
        }
        Controls.newDropdownList = newDropdownList;
        function newSlider(viewer, id, minValue, maxValue, step, selectedValue, minValueLabel, maxValueLabel, // if null, minValue and maxValue will be shown; if '', no label will be shown
        onMove, onRelease, tooltip) {
            // console.log('newSlider');
            let slider = {
                base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showSlider(slider, par) },
                minValue: minValue,
                maxValue: maxValue,
                step: step,
                selectedValue: selectedValue,
                minValueLabel: minValueLabel !== null && minValueLabel !== void 0 ? minValueLabel : minValue.toString(),
                maxValueLabel: maxValueLabel !== null && maxValueLabel !== void 0 ? maxValueLabel : maxValue.toString(),
                onMove: onMove,
                onRelease: onRelease
            };
            return slider;
        }
        Controls.newSlider = newSlider;
        function showSlider(slider, parentDiv) {
            slider.base.div = parentDiv.append('div').attr('class', 'slider').attr('id', slider.base.id);
            let sliderLeftValue = slider.base.div.append('div').attr('class', 'slider-left-value').text(slider.minValueLabel);
            let sliderRightValue = slider.base.div.append('div').attr('class', 'slider-right-value').text(slider.maxValueLabel);
            let sliderMain = slider.base.div.append('div').attr('class', 'slider-main');
            let sliderRailRuler = sliderMain.append('div').attr('class', 'slider-rail-ruler');
            sliderRailRuler.append('div').attr('class', 'slider-rail');
            let sliderRailClickable = sliderRailRuler.append('div').attr('class', 'slider-rail-clickable');
            let sliderHandleRuler = sliderRailRuler.append('div').attr('class', 'slider-handle-ruler');
            let sliderHandle = sliderHandleRuler.append('div').attr('class', 'slider-handle');
            let relativePosition = sliderValueToRelativePosition(slider, slider.selectedValue);
            sliderHandleRuler.style('left', relativePosition * 100 + '%');
            Drawing.setTooltips(slider.base.viewer, sliderMain, [slider.base.tooltip], false, true);
            Drawing.setTooltips(slider.base.viewer, sliderLeftValue, ['Click/hold to decrease value.'], false, true);
            Drawing.setTooltips(slider.base.viewer, sliderRightValue, ['Click/hold to increase value.'], false, true);
            let dragHandler = d3.drag()
                .on('start.slider', () => moveSlider(slider, d3.event.sourceEvent))
                .on('drag.slider', () => moveSlider(slider, d3.event.sourceEvent))
                .on('end.slider', () => releaseSlider(slider, d3.event.sourceEvent));
            dragHandler(sliderRailClickable);
            dragHandler(sliderHandle);
            Drawing.addMouseHoldBehavior(sliderLeftValue, () => sliderStep(slider, -1), () => sliderStep(slider, -1), () => { });
            Drawing.addMouseHoldBehavior(sliderRightValue, () => sliderStep(slider, +1), () => sliderStep(slider, +1), () => { });
        }
        Controls.showSlider = showSlider;
        function sliderStep(slider, nSteps) {
            let value = Geometry.constrain(slider.selectedValue + nSteps * slider.step, slider.minValue, slider.maxValue);
            let relative = sliderValueToRelativePosition(slider, value);
            slider.base.div.select('div.slider-handle-ruler').style('left', relative * 100 + '%');
            slider.selectedValue = value;
            slider.onMove(value);
            slider.onRelease(value);
        }
        function sliderValueToRelativePosition(slider, value) {
            let relativePosition = (value - slider.minValue) / (slider.maxValue - slider.minValue);
            relativePosition = Geometry.constrain(relativePosition, 0, 1);
            return relativePosition;
        }
        function sliderRelativePositionToValue(slider, relativePosition) {
            return slider.minValue + relativePosition * (slider.maxValue - slider.minValue);
        }
        function discretizeSliderValue(slider, value) {
            let nSteps = Math.round((value - slider.minValue) / slider.step);
            return slider.minValue + nSteps * slider.step;
        }
        function getRelativePositionOnSlider(slider, event) {
            let ruler = slider.base.div.selectAll('div.slider-rail-ruler').node();
            if (ruler !== null) {
                let { x, width } = ruler.getBoundingClientRect();
                let relativePosition = (event.clientX - x) / width;
                relativePosition = Geometry.constrain(relativePosition, 0, 1);
                return relativePosition;
            }
            else {
                return null;
            }
        }
        function moveSlider(slider, event) {
            let relativePosition = getRelativePositionOnSlider(slider, event);
            if (relativePosition !== null) {
                let value = sliderRelativePositionToValue(slider, relativePosition);
                value = discretizeSliderValue(slider, value);
                relativePosition = sliderValueToRelativePosition(slider, value);
                slider.base.div.select('div.slider-handle-ruler').style('left', relativePosition * 100 + '%');
                slider.onMove(value);
            }
        }
        function releaseSlider(slider, event) {
            let relativePosition = getRelativePositionOnSlider(slider, event);
            if (relativePosition !== null) {
                let value = sliderRelativePositionToValue(slider, relativePosition);
                value = discretizeSliderValue(slider, value);
                relativePosition = sliderValueToRelativePosition(slider, value);
                slider.base.div.select('div.slider-handle-ruler').style('left', relativePosition * 100 + '%');
                slider.selectedValue = value;
                slider.onRelease(value);
            }
        }
        function newPopupSlider(viewer, id, textPrefix, textSuffix, minValue, maxValue, step, selectedValue, minValueLabel, maxValueLabel, // if null, minValue and maxValue will be shown; if '', no label will be shown
        onMove, onRelease, tooltip, autocollapse = true) {
            let popup = newPopup(viewer, id, textPrefix + selectedValue + textSuffix, autocollapse, tooltip);
            let slider = newSlider(viewer, id, minValue, maxValue, step, selectedValue, minValueLabel, maxValueLabel, value => { changePopupText(popup, textPrefix + value + textSuffix); onMove(value); }, value => { changePopupText(popup, textPrefix + value + textSuffix); onRelease(value); }, null);
            addToPopup(popup, slider);
            return popup;
        }
        Controls.newPopupSlider = newPopupSlider;
    })(Controls || (Controls = {}));

    var __awaiter$1 = (undefined && undefined.__awaiter) || function (thisArg, _arguments, P, generator) {
        function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    };
    var OverProtViewerCore;
    (function (OverProtViewerCore) {
        function initializeViewer(htmlElement) {
            var _a;
            let settings = Types.newSettingsFromHTMLElement(htmlElement);
            const id = htmlElement.id || '';
            const uniqueId = `${id}-${Math.random().toString(36).slice(2)}`;
            console.log(`Initializing OverProt viewer with id="${id}", file="${settings.file}"`, settings);
            const d3viewer = d3.select(htmlElement);
            // d3.select(htmlElement).selectAll(()=>htmlElement.childNodes as any).remove();
            d3viewer.selectAll(':scope > *').remove(); // clear all children
            let d3mainDiv = d3viewer.append('div').attr('class', 'overprot-viewer').attr('id', 'overprot-viewer-' + uniqueId);
            let d3guiDiv = d3mainDiv.append('div').attr('class', 'gui')
                .styles({ width: settings.width, height: settings.height });
            let realSize = (_a = d3guiDiv.node()) === null || _a === void 0 ? void 0 : _a.getBoundingClientRect();
            if (realSize !== undefined) {
                let { width, height } = realSize;
                settings.width = width;
                settings.height = height;
            }
            let d3canvasDiv = d3guiDiv.append('div').attr('class', 'canvas');
            let d3canvas = d3canvasDiv.append('svg').attr('class', 'canvas')
                .attrs({ width: settings.width, height: settings.height });
            const viewerCSS = getOverProtViewerCSS();
            d3canvas.append('defs').append('style').attr('type', 'text/css').html(viewerCSS); // For rendering to PNG
            let viewer = Types.newViewer(id, uniqueId, d3viewer, d3mainDiv, d3guiDiv, d3canvas, settings);
            // initializeExternalControls(viewer);
            initializeInternalControls(viewer);
            d3mainDiv.append('br');
            d3canvas.on('dblclick', () => Drawing.zoomAll(viewer));
            d3canvas.on('wheel', () => {
                if (d3.event.ctrlKey) {
                    let ratio = d3.event.deltaY > 0 ? Constants.ZOOM_STEP_RATIO_MOUSE : 1 / Constants.ZOOM_STEP_RATIO_MOUSE;
                    Drawing.zoomOut(viewer, ratio, d3.mouse(d3canvas.node()));
                    d3.event.preventDefault();
                }
                else {
                    let messageBackground = viewer.canvas.selectAll('text.temporary-central-message').data([1])
                        .enter().append('rect').attr('class', 'temporary-central-message-background')
                        .attrs({ x: 0, width: viewer.screen.width, y: 0, height: viewer.screen.height });
                    let message = viewer.canvas.selectAll('text.temporary-central-message').data([1])
                        .enter().append('text').attr('class', 'temporary-central-message')
                        .attrs({ x: viewer.screen.width / 2, y: viewer.screen.height / 2 })
                        .text('Use Ctrl + scroll to zoom');
                    Drawing.fadeIn(messageBackground);
                    Drawing.fadeIn(message);
                    Drawing.fadeOutRemove(messageBackground, Constants.TEMPORARY_CENTRAL_MESSAGE_TIMEOUT);
                    Drawing.fadeOutRemove(message, Constants.TEMPORARY_CENTRAL_MESSAGE_TIMEOUT);
                }
            });
            let dragHandler = d3.drag()
                .on('drag', () => Drawing.shiftByMouse(viewer, -d3.event.dx, -d3.event.dy));
            dragHandler(d3canvas);
            d3.select(window).on('resize.' + uniqueId, () => {
                let newSize = d3guiDiv.node().getBoundingClientRect();
                resizeVisualization(viewer, newSize.width, newSize.height);
            });
            if (settings.listenEvents) {
                htmlElement.addEventListener(Constants.EVENT_PREFIX + Constants.EVENT_TYPE_DO_SELECT, (e) => Drawing.handleEvent(viewer, e)); // does not work with d3 .on() because of special meaning of dots in event type
                htmlElement.addEventListener(Constants.EVENT_PREFIX + Constants.EVENT_TYPE_DO_HOVER, (e) => Drawing.handleEvent(viewer, e));
            }
            viewer.canvas.append('text').attr('class', 'central-message')
                .attrs({ x: viewer.screen.width / 2, y: viewer.screen.height / 2 })
                .text('Loading...');
            if (settings.file != '') {
                let absoluteFilePath = new URL(settings.file, document.baseURI).toString();
                // console.log(`Fetching ${absoluteFilePath}`);
                fetch(absoluteFilePath)
                    .catch((error) => __awaiter$1(this, void 0, void 0, function* () {
                    console.error(`OverProt Viewer with id="${viewer.id}" failed to initialize because the request for`, absoluteFilePath, 'failed (this might be caused by CORS policy). Original error:', error);
                    let data = Dag.newDagWithError(`Failed to fetch data from "${absoluteFilePath}"`);
                    setDataToViewer(viewer, data);
                }))
                    .then((response) => __awaiter$1(this, void 0, void 0, function* () {
                    if (response && response.ok) {
                        let text = yield response.text();
                        let data = Dag.dagFromJson(text);
                        setDataToViewer(viewer, data);
                    }
                    else if (response && !response.ok) {
                        console.error(`OverProt Viewer with id="${viewer.id}" failed to initialize because the request for`, absoluteFilePath, 'returned status code', response.status, response.statusText);
                        let data = Dag.newDagWithError(`Failed to fetch data from "${absoluteFilePath}"`);
                        setDataToViewer(viewer, data);
                    }
                }))
                    .catch((error) => __awaiter$1(this, void 0, void 0, function* () {
                    console.error(`OverProt Viewer with id="${viewer.id}" failed to initialize because of internal error.`);
                    let data = Dag.newDagWithError(`Internal error`);
                    setDataToViewer(viewer, data);
                    throw error;
                }));
            }
            else {
                let data = Dag.newDagWithError('No file specified');
                setDataToViewer(viewer, data);
            }
        }
        OverProtViewerCore.initializeViewer = initializeViewer;
        function getOverProtViewerCSS() {
            var _a;
            const VIEWER = 'div.overprot-viewer';
            let styleLines = [];
            for (const sheet of document.styleSheets) {
                var rules = (_a = sheet.cssRules) !== null && _a !== void 0 ? _a : [];
                for (const rule of rules) {
                    if (rule.cssText.includes(VIEWER)) {
                        let ruleText = rule.cssText.replace(VIEWER, '');
                        styleLines.push(ruleText);
                    }
                }
            }
            return styleLines.join('\n');
        }
        function initializeInternalControls(viewer) {
            let controlsDiv = viewer.guiDiv.append('div').attr('class', 'internal-controls');
            let colorOptions = [
                ['Uniform', Enums.ColorMethod.Uniform, 'Show all SSEs in the same color.'],
                ['Type', Enums.ColorMethod.Type, 'Show &beta;-strands in blue, helices in gray.'],
                ['Sheet', Enums.ColorMethod.Sheet, 'Assign the same color to &beta;-strands from the same &beta;-sheet, <br>show helices in gray.'],
                ['Variability', Enums.ColorMethod.Stdev, '<strong>3D variability</strong> measures the standard deviation of the SSE end point coordinates.<br>Low values (dark) indicate conserved SSE position, <br>high values (bright) indicate variable SSE position.'],
                ['Rainbow', Enums.ColorMethod.Rainbow, 'Rainbow coloring from N-terminus (blue) to C-terminus (red).'],
            ];
            let shapeOptions = [
                ['Rectangle', Enums.ShapeMethod.Rectangle, 'Show SSEs as rectangles. <br>Height of the rectangle indicates <strong>occurrence</strong> (what percentage of structures contain this SSE), <br>width indicates <strong>average length</strong> (number of residues).'],
                ['SymCDF', Enums.ShapeMethod.SymCdf, '<strong>Cumulative distribution function</strong> describes the statistical distribution of the SSE length. <br>The widest part of the shape corresponds to maximum length, the narrowest to minimum length, <br> the height corresponds to occurrence. <br>(The SymCDF shape consists of four symmetrical copies of the CDF, the bottom right quarter is the classical CDF.)'],
            ];
            let connectivityOptions = [
                ['On', true, null],
                ['Off', false, null],
            ];
            let controlPanel = Controls.newControlPanel(viewer, 'main-panel', null);
            // let zoomInButton = Controls.newButton(viewer, 'zoom-in', '+', true, false, () => Drawing.zoomIn(viewer), 'Zoom in. <br>Tip: Use Ctrl + mouse wheel to zoom.');
            // Controls.addToControlPanel(controlPanel, zoomInButton);
            // let zoomOutButton = Controls.newButton(viewer, 'zoom-out', '-', true, false, () => Drawing.zoomOut(viewer), 'Zoom out. <br>Tip: Use Ctrl + mouse wheel to zoom.');
            // Controls.addToControlPanel(controlPanel, zoomOutButton);
            // let zoomResetButoon = Controls.newButton(viewer, 'zoom-reset', Constants.RESET_SYMBOL, true, false, () => Drawing.zoomAll(viewer), 'Reset zoom. <br>Tip: Use double click to reset zoom.');
            // Controls.addToControlPanel(controlPanel, zoomResetButoon);
            let zoomInButton2 = Controls.newButton(viewer, 'zoom-in', Constants.ICON_PLUS, true, true, () => Drawing.zoomIn(viewer), 'Zoom in. <br>Tip: Use Ctrl + mouse wheel to zoom.');
            Controls.addToControlPanel(controlPanel, zoomInButton2);
            let zoomOutButton2 = Controls.newButton(viewer, 'zoom-out', Constants.ICON_MINUS, true, true, () => Drawing.zoomOut(viewer), 'Zoom out. <br>Tip: Use Ctrl + mouse wheel to zoom.');
            Controls.addToControlPanel(controlPanel, zoomOutButton2);
            let zoomResetButoon2 = Controls.newButton(viewer, 'zoom-reset', Constants.ICON_RESET, true, true, () => Drawing.zoomAll(viewer), 'Reset zoom. <br>Tip: Use double click to reset zoom.');
            Controls.addToControlPanel(controlPanel, zoomResetButoon2);
            let colorMethodDropdown = Controls.newDropdownList(viewer, 'color-method', 'Color', colorOptions, viewer.settings.colorMethod, method => applyColors(viewer, method), 'Choose coloring method.', true, false);
            Controls.addToControlPanel(controlPanel, colorMethodDropdown);
            let shapeMethodDropdown = Controls.newDropdownList(viewer, 'shape-method', 'Shape', shapeOptions, viewer.settings.shapeMethod, method => applyShapes(viewer, method), 'Choose shape method.', true, false);
            Controls.addToControlPanel(controlPanel, shapeMethodDropdown);
            let betaConnectivityDropdown = Controls.newDropdownList(viewer, 'beta-connectivity', 'Beta-connectivity', connectivityOptions, viewer.settings.betaConnectivityVisibility, method => Drawing.showBetaConnectivity(viewer, method, true), '<strong>Beta-connectivity</strong> shows how &beta;-strands are connected to each other in &beta;-sheets.<br>Upper arcs indicate antiparallel connections,<br>lower arcs indicate parallel connections.', true, false);
            Controls.addToControlPanel(controlPanel, betaConnectivityDropdown);
            let occurrenceThresholdSlider = Controls.newPopupSlider(viewer, 'occurrence-threshold', 'Occurrence threshold: ', '%', 0, 100, 1, viewer.settings.occurrenceThreshold * 100, '0%', '100%', val => { }, val => applyFiltering(viewer, val / 100), 'Hide SSEs with occurrence lower than the specified threshold.');
            Controls.addToControlPanel(controlPanel, occurrenceThresholdSlider);
            // let saveButton = Controls.newButton(viewer, 'save', '&#10515;', true, false, () => Drawing.save(viewer), 'Save image.');
            // Controls.addToControlPanel(controlPanel, saveButton);
            let saveButton2 = Controls.newButton(viewer, 'save', Constants.ICON_CAMERA, true, true, () => Drawing.save(viewer), 'Save image.');
            Controls.addToControlPanel(controlPanel, saveButton2);
            controlPanel.base.show(controlsDiv);
        }
        function setDataToViewer(viewer, data) {
            viewer.data = data;
            Dag.filterDagAndAddLevels(viewer.data, viewer.settings.occurrenceThreshold);
            refreshVisualization(viewer);
        }
        function refreshVisualization(viewer) {
            // viewer.canvas.selectAll('*').remove();
            viewer.canvas.selectAll(':scope > :not(defs)').remove(); // clear all children but defs
            if (viewer.data.error !== null) {
                viewer.canvas.append('text').attr('class', 'central-message')
                    .attrs({ x: viewer.screen.width / 2, y: viewer.screen.height / 2 })
                    .text('Error: ' + viewer.data.error);
                return;
            }
            if (viewer.data.activeNodes.length == 0) {
                viewer.canvas.append('text').attr('class', 'central-message')
                    .attrs({ x: viewer.screen.width / 2, y: viewer.screen.height / 2 })
                    .text('No secondary structure elements');
                return;
            }
            let dag = viewer.data;
            let reprLengths = viewer.settings.shapeMethod == Enums.ShapeMethod.SymCdf ?
                dag.nodes.map(node => node.cdf[node.cdf.length - 1][0] * Constants.LENGTH_SCALE)
                : dag.nodes.map(node => node.avg_length * Constants.LENGTH_SCALE);
            let reprHeights = dag.nodes.map(node => node.occurrence * Constants.OCCURRENCE_SCALE);
            let reprWeights = dag.nodes.map(node => node.occurrence * node.avg_length);
            if (viewer.settings.layoutMethod == Enums.LayoutMethod.Old) {
                let nNodes = dag.nodes.length;
                let nLevels = dag.levels.length;
                let starts = new Array(nNodes);
                let ends = new Array(nNodes);
                let floors = new Array(nNodes).fill(0);
                let currentX = 0;
                for (let iLevel = 0; iLevel < nLevels; iLevel++) {
                    const level = dag.levels[iLevel];
                    let nFloors = level.length;
                    for (let iFloor = 0; iFloor < nFloors; iFloor++) {
                        const iNode = level[iFloor];
                        starts[iNode] = currentX;
                        ends[iNode] = currentX + reprLengths[iNode];
                        floors[iNode] = iFloor - (nFloors - 1) / 2;
                    }
                    currentX = Math.max(...level.map(v => ends[v])) + Constants.GAP_LENGTH;
                }
                for (let iNode = 0; iNode < dag.nodes.length; iNode++) {
                    dag.nodes[iNode].visual = Dag.newNodeVisual();
                    if (dag.nodes[iNode].active) {
                        dag.nodes[iNode].visual.rect = {
                            x: starts[iNode],
                            width: reprLengths[iNode],
                            y: floors[iNode] * Constants.FLOOR_HEIGHT - reprHeights[iNode] / 2,
                            height: reprHeights[iNode]
                        };
                    }
                    else { // put the inactive nodes somewhere, where they don't cause problems
                        dag.nodes[iNode].visual.rect = {
                            x: -Constants.LEFT_MARGIN,
                            width: 0,
                            y: 0,
                            height: 0
                        };
                    }
                }
                let totalLength = currentX - Constants.GAP_LENGTH;
                let minFloor = Math.min(...floors);
                let maxFloor = Math.max(...floors);
                let minX = -Constants.LEFT_MARGIN;
                let maxX = totalLength + Constants.RIGHT_MARGIN;
                let minY = minFloor * Constants.FLOOR_HEIGHT - 0.5 * Constants.OCCURRENCE_SCALE - Constants.TOP_MARGIN;
                let maxY = maxFloor * Constants.FLOOR_HEIGHT + 0.5 * Constants.OCCURRENCE_SCALE + Constants.BOTTOM_MARGIN;
                viewer.world = { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
            }
            else if (viewer.settings.layoutMethod == Enums.LayoutMethod.New) {
                let graph = Graphs.newDagFromPrecedence(dag.levels, dag.precedence);
                let vertexSizes = new Map();
                for (const v of graph.vertices)
                    vertexSizes.set(v, { width: reprLengths[v], height: reprHeights[v], weight: reprWeights[v] });
                const [box, positions] = Graphs.embedDag(graph, vertexSizes, { x: Constants.GAP_LENGTH, y: Constants.FLOOR_HEIGHT - Constants.OCCURRENCE_SCALE }, Constants.LEFT_MARGIN, Constants.RIGHT_MARGIN, Constants.TOP_MARGIN, Constants.BOTTOM_MARGIN);
                viewer.world = { x: -box.left, y: -box.top, width: box.right + box.left, height: box.bottom + box.top };
                for (let iNode = 0; iNode < dag.nodes.length; iNode++) {
                    dag.nodes[iNode].visual = Dag.newNodeVisual();
                    if (dag.nodes[iNode].active) {
                        const { x, y } = positions.get(iNode);
                        const { width, height } = vertexSizes.get(iNode);
                        dag.nodes[iNode].visual.rect = { x: x - width / 2, y: y - height / 2, width: width, height: height };
                    }
                    else { // put the inactive nodes somewhere, where they don't cause problems
                        dag.nodes[iNode].visual.rect = { x: viewer.world.x, y: viewer.world.y, width: 0, height: 0 };
                    }
                }
            }
            let initialZoomout = Geometry.zoomAllZoomout(viewer.world, viewer.screen);
            let minYZoomout = (Constants.OCCURRENCE_SCALE + Constants.TOP_MARGIN + Constants.BOTTOM_MARGIN) / viewer.screen.height;
            let maxYZoomout = viewer.world.height / viewer.screen.height;
            let minXZoomout = minYZoomout / 10; //25;
            let maxXZoomout = initialZoomout * Constants.MAX_X_ZOOMOUT;
            viewer.zoom = Geometry.newZoomInfo(minXZoomout, maxXZoomout, minYZoomout, maxYZoomout, initialZoomout);
            viewer.visWorld = Geometry.centeredVisWorld(viewer.world, viewer.screen, viewer.zoom);
            dag.precedenceLines = [];
            for (const edge of dag.precedence) {
                let u = dag.nodes[edge[0]].visual.rect;
                let v = dag.nodes[edge[1]].visual.rect;
                let xu = u.x + u.width;
                let yu = u.y + u.height / 2;
                let xv = v.x;
                let yv = v.y + v.height / 2;
                dag.precedenceLines.push({ x1: xu, y1: yu, x2: xu + Constants.KNOB_LENGTH, y2: yu });
                dag.precedenceLines.push({ x1: xu + Constants.KNOB_LENGTH, y1: yu, x2: xv - Constants.KNOB_LENGTH, y2: yv });
                dag.precedenceLines.push({ x1: xv - Constants.KNOB_LENGTH, y1: yv, x2: xv, y2: yv });
            }
            let d3nodes = viewer.canvas
                .append('g').attr('class', 'nodes')
                .selectAll('g.node')
                .data(dag.nodes)
                .enter()
                .append('g').attr('class', 'node')
                .attr('opacity', n => n.active ? 1 : 0);
            viewer.nodeMap = new Map();
            d3nodes.each((d, i, nodes) => viewer.nodeMap.set(d.label, nodes[i]));
            let d3nodeShapes;
            if (viewer.settings.shapeMethod == Enums.ShapeMethod.SymCdf) {
                d3nodeShapes = d3nodes.append('polygon');
            }
            else {
                d3nodeShapes = d3nodes.append('rect');
            }
            let d3activeShapes = d3nodeShapes.filter(n => n.active);
            Drawing.addPointBehavior(viewer, d3nodeShapes, shape => Drawing.selectNodeFromShape(viewer, shape, true), nodes => Drawing.dispatchMixedEvent(viewer, Constants.EVENT_TYPE_HOVER, nodes.data()));
            Drawing.addPickBehavior(viewer, d3nodeShapes, shape => Drawing.selectNodeFromShape(viewer, shape, true), nodes => Drawing.dispatchMixedEvent(viewer, Constants.EVENT_TYPE_SELECT, nodes.data()));
            Drawing.setTooltips(viewer, d3nodeShapes, d3nodes.data().map(createNodeTooltip), true, false);
            d3nodes
                .append('text').attr('class', 'node-label')
                .text(n => n.label);
            viewer.canvas
                .append('g').attr('class', 'edges')
                .selectAll()
                .data(dag.precedenceLines)
                .enter()
                .append('line').attr('stroke', Constants.EDGE_COLOR);
            Drawing.showBetaConnectivity(viewer, viewer.settings.betaConnectivityVisibility, false);
            applyColors(viewer, viewer.settings.colorMethod, false);
            Drawing.zoomAll(viewer);
            Drawing.redraw(viewer, false);
        }
        function resizeVisualization(viewer, newWidth, newHeight) {
            let oldZoomChanged = viewer.zoom.zoomout != viewer.zoom.initialZoomout;
            let oldZoomout = viewer.zoom.zoomout;
            let oldVisWorld = viewer.visWorld;
            let newZoomout = oldZoomout * viewer.screen.width / newWidth;
            viewer.canvas.attr('width', newWidth).attr('height', newHeight);
            viewer.screen.width = newWidth;
            viewer.screen.height = newHeight;
            refreshVisualization(viewer);
            if (oldZoomChanged) {
                Drawing.zoomSet(viewer, newZoomout, Geometry.rectangleCenter(oldVisWorld));
            }
        }
        function createNodeTooltip(node) {
            if (node.active) {
                let [minLength, maxLength] = Dag.getNodeMinMaxLength(node);
                return [
                    '<div class="node-tooltip">',
                    `<span class="tooltip-title">${node.label}</span>`,
                    '<table>',
                    `<tr><td>Type:</td><td>${node.type == 'e' ? 'Strand' : 'Helix'}</td></tr>`,
                    `<tr><td>Occurrence:</td><td>${Geometry.round(node.occurrence * 100, 1)}%</td></tr>`,
                    `<tr><td>Min. length:</td><td>${minLength} residues</td></tr>`,
                    `<tr><td>Avg. length:</td><td>${Geometry.round(node.avg_length, 2)} residues</td></tr>`,
                    `<tr><td>Max. length:</td><td>${maxLength} residues</td></tr>`,
                    `<tr><td>3D variability:</td><td>${Geometry.round(node.stdev3d, 2)} &#8491;</td></tr>`,
                    node.type == 'e' ? `<tr><td>Sheet:</td><td>${node.sheet_id}</td></tr>` : '',
                    '</table>',
                    '</div>'
                ].join('');
            }
            else {
                return null;
            }
        }
        function applyFiltering(viewer, occurrenceThreshold) {
            viewer.settings.occurrenceThreshold = occurrenceThreshold;
            Dag.filterDagAndAddLevels(viewer.data, occurrenceThreshold);
            refreshVisualization(viewer);
        }
        function applyColors(viewer, colorBy, transition = true) {
            viewer.settings.colorMethod = colorBy;
            viewer.data.nodes.forEach((n, i) => {
                let col;
                switch (colorBy) {
                    case Enums.ColorMethod.Uniform:
                        col = Colors.NEUTRAL_COLOR;
                        break;
                    case Enums.ColorMethod.Type:
                        col = Colors.bySseType(n.type);
                        break;
                    case Enums.ColorMethod.Sheet:
                        col = Colors.byIndex1(n.sheet_id);
                        break;
                    case Enums.ColorMethod.Stdev:
                        col = Colors.byExpHeatmap(n.stdev3d, Constants.HEATMAP_MIDDLE_VALUE);
                        break;
                    case Enums.ColorMethod.Rainbow:
                        col = (n.rainbow_hex != undefined) ? d3.rgb(n.rainbow_hex) : Colors.NEUTRAL_COLOR;
                        break;
                }
                n.visual.fill = col.hex();
                n.visual.stroke = col.darker().hex();
            });
            Drawing.recolor(viewer, transition);
        }
        function applyShapes(viewer, shapeMethod) {
            viewer.settings.shapeMethod = shapeMethod;
            setDataToViewer(viewer, viewer.data);
        }
    })(OverProtViewerCore || (OverProtViewerCore = {}));

    (function (OverProtViewer) {
        class OverProtViewerElement extends HTMLElement {
            constructor() {
                super();
            }
            connectedCallback() {
                this.initialize();
            }
            /** Initialize this overprot-viewer element using its HTML attributes.
             * Return true on success, false on failure to load data.
            */
            initialize() {
                return OverProtViewerCore.initializeViewer(this);
            }
        }
        window.customElements.define('overprot-viewer', OverProtViewerElement);
    })();

}(d3));
