var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import * as d3 from 'd3';
import { Constants } from './Constants';
import { Colors } from './Colors';
import { Geometry } from './Geometry';
import { Enums } from './Enums';
export var Drawing;
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
            .selectAll('path');
        if (viewer.settings.colorMethod != Enums.ColorMethod.Stdev) {
            betaArcs.style('stroke', ladder => viewer.data.nodes[ladder[0]].visual.stroke);
        }
        else {
            let arcColor = Colors.NEUTRAL_DARK.hex();
            betaArcs.style('stroke', arcColor);
        }
        show3DVariabilityLegend(viewer, viewer.settings.colorMethod == Enums.ColorMethod.Stdev, transition);
    }
    Drawing.recolor = recolor;
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
        d3nodes.select('text')
            .transition().duration(duration)
            .attrs(n => {
            let { x, y, height, width } = Geometry.rectToScreen(viewer.visWorld, viewer.screen, n.visual.rect);
            return { x: x + width / 2, y: y + height + Constants.HANGING_TEXT_OFFSET };
        })
            .style('opacity', n => nodeBigEnoughForLabel(viewer, n) ? 1 : 0);
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
        // .attr('d', edge => {
        //     if ((edge as Number[])[3] == 0) return '';
        //     let [u, v, orientation] = edge as number[];
        //     let n1 = viewer.data.nodes[u].visual.rect;
        //     let n2 = viewer.data.nodes[v].visual.rect;
        //     let endpoints = Geometry.lineToScreen(viewer.visWorld, viewer.screen, { x1: n1.x + 0.5 * n1.width, y1: n1.y + 0.5 * n1.height, x2: n2.x + 0.5 * n2.width, y2: n2.y + 0.5 * n2.height });
        //     let x2yZoomRatio = viewer.zoom.yZoomout / viewer.zoom.xZoomout;
        //     // return Geometry.arcPathD_circle(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
        //     return Geometry.arcPathD_ellipse(endpoints, viewer.world.width / viewer.zoom.xZoomout, arcMaxMinor, orientation == 1, x2yZoomRatio);
        //     // return Geometry.arcPathD_bezier(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
        // });
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
    function gradientBar(canvas, rect) {
        let bar = canvas.append('g');
        let n = Math.floor(rect.width);
        let step = (rect.width - 1) / n;
        bar.selectAll('rect.fill')
            .data(d3.range(n))
            .enter()
            .append('rect')
            .attrs(i => {
            return {
                class: 'fill', x: rect.x + i * step, y: rect.y, width: step + 1, height: rect.height,
                stroke: 'none', fill: Colors.byLinHeatmap(i, n - 1).hex()
            };
        });
        bar.append('rect')
            .attrs(i => { return Object.assign(Object.assign({ class: 'border' }, rect), { fill: 'none' }); });
        return bar;
    }
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
            let betaConnectivityPaths = betaConnectivityVis.selectAll()
                .data(dag.beta_connectivity)
                .enter()
                .append('path')
                .style('stroke', ladder => dag.nodes[ladder[0]].visual.stroke);
            addPointBehavior(viewer, betaConnectivityPaths);
            redraw(viewer, false);
            if (transition) {
                fadeIn(betaConnectivityVis);
            }
        }
    }
    Drawing.showBetaConnectivity = showBetaConnectivity;
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
    function dispatchSseEvent(viewer, eventType, sses) {
        var _a;
        if (!viewer.settings.dispatchEvents)
            return;
        let eventDetail = {
            sourceType: (_a = viewer.d3viewer.node()) === null || _a === void 0 ? void 0 : _a.tagName,
            sourceId: viewer.id,
            sourceInternalId: viewer.uniqueId,
            eventType: eventType,
            targetType: 'sses',
            sses: sses.map(sse => {
                return {
                    label: sse.label,
                    type: sse.type,
                    sheetId: sse.type.toLowerCase() == 'e' ? sse.sheet_id : null,
                };
            }),
        };
        viewer.mainDiv.dispatch(Constants.EVENT_PREFIX + eventType, { detail: eventDetail, bubbles: true });
    }
    Drawing.dispatchSseEvent = dispatchSseEvent;
    function handleEvent(viewer, event) {
        var _a;
        // console.log('Inbound event', event.type, event);
        const detail = event.detail;
        if (detail == null || detail == undefined) {
            console.error(`Event ${event.type}: event.detail must be an object.`);
            return;
        }
        if (detail.sourceType == ((_a = viewer.d3viewer.node()) === null || _a === void 0 ? void 0 : _a.tagName) && detail.sourceInternalId == viewer.uniqueId) {
            console.log('Ignoring self', viewer.uniqueId);
            return;
        }
        const sses = detail.sses;
        if (sses == undefined) {
            console.error(`Event ${event.type}: event.detail.sses must be an array.`);
            return;
        }
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
        viewer.canvas.selectAll(`g.node[${attribute}]`).attr(attribute, null);
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
        if (event.type == PDB_OVERPROT_DO_SELECT) {
            unpinAllTooltips(viewer);
        }
    }
    Drawing.handleEvent = handleEvent;
})(Drawing || (Drawing = {}));
//# sourceMappingURL=Drawing.js.map