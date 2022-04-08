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
import { Types } from './Types';
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
        showLegend(viewer, transition);
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
    function gradientBarExp(parentNode, rect, maxValue, middle, ticksDistance, legend, ticksAbove, labelLeft) {
        let bar = parentNode.append('g').attr('class', 'heatmap-bar')
            .attr('transform', `translate(${rect.x},${rect.y})`);
        let tickTextY = ticksAbove ? -Constants.HANGING_TEXT_OFFSET : (rect.height + Constants.HANGING_TEXT_OFFSET);
        let tickTextDomBaseline = ticksAbove ? 'alphabetic' : 'hanging';
        let labelX = labelLeft ? -5 : (rect.width + 5);
        let labelAnchor = labelLeft ? 'end' : 'start';
        let barLabel = bar.append('text').attr('class', 'heatmap-bar-label')
            .attrs({ x: labelX, y: 0.5 * rect.height })
            .style('text-anchor', labelAnchor)
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
            .attr('y', tickTextY)
            .style('dominant-baseline', tickTextDomBaseline)
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
    const LEGEND_BAR_WIDTH = 200;
    const LEGEND_BAR_HEIGHT = 18;
    const LEGEND_HMARGIN = 15;
    const LEGEND_VMARGIN = 5;
    const LEGEND_SPACING = 10; // between items
    const LEGEND_SPACING_INNER = 5; // between item shape and label
    function showLegend(viewer, transition = true) {
        fadeOutRemove(viewer.canvas.selectAll('g.legend'));
        let legendGroup = viewer.canvas.append('g').attr('class', 'legend');
        if (viewer.settings.betaConnectivityVisibility) {
            showBetaConnectivityLegend(viewer, legendGroup, transition);
        }
        switch (viewer.settings.colorMethod) {
            case Enums.ColorMethod.Stdev:
                show3DVariabilityLegend(viewer, legendGroup, transition);
                break;
            case Enums.ColorMethod.Type:
                showTypeLegend(viewer, legendGroup, transition);
                break;
            case Enums.ColorMethod.Sheet:
                showSheetLegend(viewer, legendGroup, transition);
                break;
        }
        let w = legendGroup.node().getBBox().width;
        let x = viewer.screen.width - LEGEND_HMARGIN - w;
        let y = viewer.screen.height - LEGEND_VMARGIN - LEGEND_BAR_HEIGHT;
        legendGroup.attr('transform', `translate(${x},${y})`);
        if (transition) {
            fadeIn(legendGroup);
        }
    }
    function show3DVariabilityLegend_Old(viewer, legendGroup, transition = true) {
        var _a, _b, _c, _d;
        let bar = gradientBarExp(legendGroup, { x: viewer.screen.width - LEGEND_BAR_WIDTH - LEGEND_HMARGIN, y: LEGEND_VMARGIN, width: LEGEND_BAR_WIDTH, height: LEGEND_BAR_HEIGHT }, 15, 5, 5, '3D variability [\u212B]', false, true);
        let controlsRight = (_b = (_a = viewer.mainDiv.select('div.control-panel#main-panel').node()) === null || _a === void 0 ? void 0 : _a.getBoundingClientRect()) === null || _b === void 0 ? void 0 : _b.right;
        let barLeft = (_d = (_c = bar.node()) === null || _c === void 0 ? void 0 : _c.getBoundingClientRect()) === null || _d === void 0 ? void 0 : _d.left;
        if (controlsRight !== undefined && barLeft !== undefined && controlsRight > barLeft) {
            moveGradientBarExp(bar, viewer.screen.width - LEGEND_BAR_WIDTH - LEGEND_HMARGIN, viewer.screen.height - LEGEND_VMARGIN - LEGEND_BAR_HEIGHT, true);
        }
        if (transition) {
            fadeIn(bar);
        }
    }
    function show3DVariabilityLegend(viewer, legendGroup, transition = true) {
        let xOffset = boxWidth(legendGroup, LEGEND_SPACING);
        let bar = gradientBarExp(legendGroup, { x: xOffset, y: 0, width: LEGEND_BAR_WIDTH, height: LEGEND_BAR_HEIGHT }, 15, 5, 5, '3D variability [\u212B]', true, false);
    }
    function addLegendItem(viewer, legendGroup, rect, color, text, shape = 'rectangle') {
        let itemGroup = legendGroup.append('g').attr('class', 'legend-item');
        let x0, x1, y, r;
        switch (shape) {
            case 'rectangle':
                itemGroup.append('rect').attrs(Object.assign(Object.assign({}, rect), { stroke: color.darker().hex(), fill: color.hex() }));
                break;
            case 'upper_arc':
                x0 = rect.x;
                x1 = rect.x + rect.width;
                y = rect.y + 0.6 * rect.height;
                r = 0.5 * rect.width;
                itemGroup.append('path')
                    .attr('d', `M${x0},${y} A${r},${r} 0 0,1 ${x1},${y}`)
                    .attr('fill', 'none')
                    .attr('stroke', color.darker().hex());
                break;
            case 'lower_arc':
                x0 = rect.x;
                x1 = rect.x + rect.width;
                y = rect.y + 0.4 * rect.height;
                r = 0.5 * rect.width;
                itemGroup.append('path')
                    .attr('d', `M${x0},${y} A${r},${r} 0 0,0 ${x1},${y}`)
                    .attr('fill', 'none')
                    .attr('stroke', color.darker().hex());
                break;
        }
        let textX = rect.x + rect.width + LEGEND_SPACING_INNER;
        let textElem = itemGroup.append('text').attrs({ x: textX, y: rect.y + 0.5 * rect.height }).text(text);
        let w = textElem.node().getBBox().width;
        return textX + w + LEGEND_SPACING;
    }
    function showTypeLegend(viewer, legendGroup, transition = true) {
        let xOffset = boxWidth(legendGroup, LEGEND_SPACING);
        xOffset = addLegendItem(viewer, legendGroup, { x: xOffset, y: 0, width: LEGEND_BAR_HEIGHT, height: LEGEND_BAR_HEIGHT }, Colors.bySseType('H'), 'Helix');
        xOffset = addLegendItem(viewer, legendGroup, { x: xOffset, y: 0, width: LEGEND_BAR_HEIGHT, height: LEGEND_BAR_HEIGHT }, Colors.bySseType('E'), 'Strand');
    }
    function showSheetLegend(viewer, legendGroup, transition = true) {
        let sheetIdSet = new Set();
        for (let i of viewer.data.activeNodes) {
            let id = viewer.data.nodes[i].sheet_id;
            sheetIdSet.add(id);
        }
        sheetIdSet.delete(0);
        let sheetIds = [...sheetIdSet];
        sheetIds.sort((a, b) => a - b);
        let xOffset = boxWidth(legendGroup, LEGEND_SPACING);
        xOffset = addLegendItem(viewer, legendGroup, { x: xOffset, y: 0, width: LEGEND_BAR_HEIGHT, height: LEGEND_BAR_HEIGHT }, Colors.bySseType('H'), 'Helix');
        for (let id of sheetIds) {
            xOffset = addLegendItem(viewer, legendGroup, { x: xOffset, y: 0, width: LEGEND_BAR_HEIGHT, height: LEGEND_BAR_HEIGHT }, Colors.byIndex1(id), `\u03b2${id}`);
        }
    }
    function showBetaConnectivityLegend(viewer, legendGroup, transition = true) {
        let xOffset = boxWidth(legendGroup, LEGEND_SPACING);
        xOffset = addLegendItem(viewer, legendGroup, { x: xOffset, y: 0, width: LEGEND_BAR_HEIGHT, height: LEGEND_BAR_HEIGHT }, Colors.NEUTRAL_COLOR, 'Parallel', 'lower_arc');
        xOffset = addLegendItem(viewer, legendGroup, { x: xOffset, y: 0, width: LEGEND_BAR_HEIGHT, height: LEGEND_BAR_HEIGHT }, Colors.NEUTRAL_COLOR, 'Antiparallel', 'upper_arc');
    }
    function boxWidth(selection, spacing = 0) {
        let result = selection.node().getBBox().width;
        if (result > 0) {
            result += spacing;
        }
        return result;
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
        showLegend(viewer, transition);
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
    function concatSelections(...selections) {
        let nodes = [];
        for (const selection of selections)
            nodes.push(...selection.nodes());
        return d3.selectAll(nodes);
    }
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
        const detail = event.detail;
        if (detail == null || detail == undefined) {
            console.error(`Event ${event.type}: event.detail must be an object.`);
            return;
        }
        if (detail.sourceType == ((_a = viewer.d3viewer.node()) === null || _a === void 0 ? void 0 : _a.tagName.toLowerCase()) && detail.sourceInternalId == viewer.internalId) {
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
//# sourceMappingURL=Drawing.js.map