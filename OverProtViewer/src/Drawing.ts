import * as d3 from 'd3';
import * as d3SelectionMulti from 'd3-selection-multi';
import { Types } from './Types';
import { Constants } from './Constants';
import { Dag } from './Dag';
import { Colors } from './Colors';
import { Geometry } from './Geometry';
import { Enums } from './Enums';
import { html, D3BrushEvent } from 'd3';


export namespace Drawing {

    export function zoomOut(viewer: Types.Viewer, ratio = Constants.ZOOM_STEP_RATIO, mouseXY?: number[]): void {
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

    export function zoomIn(viewer: Types.Viewer, ratio = Constants.ZOOM_STEP_RATIO, mouseXY?: number[]): void {
        zoomOut(viewer, 1 / ratio, mouseXY);
    }

    export function zoomAll(viewer: Types.Viewer): void {
        Geometry.zoomInfoSetZoomout(viewer.zoom, viewer.zoom.initialZoomout);
        viewer.visWorld = Geometry.centeredVisWorld(viewer.world, viewer.screen, viewer.zoom);
        redraw(viewer);
        updateZoomButtons(viewer);
    }
    export function zoomSet(viewer: Types.Viewer, newZoomout: number, centerXY?: [number, number]): void {
        Geometry.zoomInfoSetZoomout(viewer.zoom, newZoomout);
        let [centerX, centerY] = centerXY ?? Geometry.rectangleCenter(viewer.visWorld);
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

    function updateZoomButtons(viewer: Types.Viewer): void {
        viewer.mainDiv.selectAll('div.button#zoom-out').attr('disabled', viewer.zoom.zoomout >= viewer.zoom.maxZoomout ? 'true' : null as any);
        viewer.mainDiv.selectAll('div.button#zoom-in').attr('disabled', viewer.zoom.zoomout <= viewer.zoom.minZoomout ? 'true' : null as any);
        viewer.mainDiv.selectAll('div.button#zoom-reset').attr('disabled', viewer.zoom.zoomout == viewer.zoom.initialZoomout ? 'true' : null as any);
    }

    export function shift(viewer: Types.Viewer, rightRelative: number, downRelative: number): void {
        let { width, height } = viewer.visWorld;
        viewer.visWorld.x += rightRelative * width;
        viewer.visWorld.y += downRelative * height;
        constrainVisWorldXY(viewer);
        redraw(viewer);
    }

    export function shiftByMouse(viewer: Types.Viewer, rightPixels: number, downPixels: number): void {
        let { width, height } = viewer.visWorld;
        let { width: sw, height: sh } = viewer.screen;
        viewer.visWorld.x += rightPixels * width / sw;
        viewer.visWorld.y += downPixels * height / sh;
        constrainVisWorldXY(viewer);
        redraw(viewer, false);
    }

    function placeTooltip(viewer: Types.Viewer, tooltip?: d3.Selection<HTMLDivElement, unknown, HTMLElement, any>): d3.Selection<HTMLDivElement, unknown, HTMLElement, any> {
        tooltip = tooltip || viewer.mainDiv.select('div.tooltip') as unknown as d3.Selection<HTMLDivElement, unknown, HTMLElement, any>;
        return tooltip
            .style('left', (d3.event.clientX + Constants.TOOLTIP_OFFSET.x))
            .style('top', (d3.event.clientY + Constants.TOOLTIP_OFFSET.y));
    }

    function tooltipMouseEnter(viewer: Types.Viewer, targetElement: HTMLElement, htmlContent: string, delay = false): void {
        let hasPinnedTooltip = d3.select(targetElement).attr('tooltip') == 'pinned';
        if (!hasPinnedTooltip) {
            let tooltip = viewer.mainDiv.append('div').attr('class', 'tooltip').attr('type', 'hover').html(htmlContent);
            placeTooltip(viewer, tooltip as any);
            d3.select(targetElement)?.attr('tooltip', 'hover');
            fadeIn(tooltip as unknown as Types.D3Selection, delay ? Constants.TOOLTIP_DELAY : 0);
        }
    }
    function tooltipMouseMove(viewer: Types.Viewer): void {
        let tooltip = viewer.mainDiv.selectAll('div.tooltip[type=hover]');
        placeTooltip(viewer, tooltip as any);
    }
    function tooltipMouseLeave(viewer: Types.Viewer): void {
        viewer.mainDiv.selectAll('div.tooltip[type=hover]').remove();
        viewer.mainDiv.selectAll('[tooltip=hover]').attr('tooltip', null);
    }
    function tooltipMouseClick(viewer: Types.Viewer, targetElement: HTMLElement, htmlContent: string): void {
        // add pre-pinned tooltip
        let tooltip = viewer.mainDiv.append('div').attr('class', 'tooltip').attr('type', 'pre-pinned').html(htmlContent);
        placeTooltip(viewer, tooltip as any);
        d3.select(targetElement)?.attr('tooltip', 'pre-pinned');
        // add listener to GUI to remove all pinned tooltips on click and remove all tooltip='pinned' attributes and change pre-pinned to pinned - will be invoked in this bubbling
        viewer.mainDiv.on('click.tooltip', () => {
            viewer.mainDiv.selectAll('div.tooltip[type=hover],div.tooltip[type=pinned]').remove();
            viewer.mainDiv.selectAll('div.tooltip[type=pre-pinned]').attr('type', 'pinned');
            viewer.mainDiv.selectAll('[tooltip=hover],[tooltip=pinned]').attr('tooltip', null);
            viewer.mainDiv.selectAll('[tooltip=pre-pinned]').attr('tooltip', 'pinned');
        });
    }
    function unpinAllTooltips(viewer: Types.Viewer): void {
        viewer.mainDiv.selectAll('div.tooltip[type=pinned]').remove();
        viewer.mainDiv.selectAll('[tooltip=pinned]').attr('tooltip', null);
    }


    export function setTooltips(viewer: Types.Viewer, selection: Types.D3Selection, htmlContents: (string | null)[] | null, pinnable = false, delay = false) {
        if (htmlContents === null) {
            selection
                .on('mouseenter.tooltip', null)
                .on('mousemove.tooltip', null)
                .on('mouseleave.tooltip', null)
                .on('click.tooltip', null);
        } else {
            let active = htmlContents.map(content => content != null);
            let activeContents = htmlContents.filter(content => content != null) as string[];
            selection.filter((d, i) => active[i])
                .on('mouseenter.tooltip', (d, i, g) => tooltipMouseEnter(viewer, g[i] as any, activeContents[i], delay))
                .on('mousemove.tooltip', (d, i) => tooltipMouseMove(viewer))
                .on('mouseleave.tooltip', (d, i) => tooltipMouseLeave(viewer));
            if (pinnable) {
                selection.filter((d, i) => active[i])
                    .on('click.tooltip', (d, i, g) => tooltipMouseClick(viewer, g[i] as any, activeContents[i]));
            }
            selection.filter((d, i) => !active[i])
                .on('mouseenter.tooltip', null)
                .on('mousemove.tooltip', null)
                .on('mouseleave.tooltip', null)
                .on('click.tooltip', null);
        }
    }

    export function addPointBehavior(viewer: Types.Viewer, selection: Types.D3Selection,
        pointedElementSelector = (pointed: HTMLElement) => (d3.select(pointed) as any as Types.D3Selection),
        callback: ((sel: Types.D3Selection) => any) | null = null) {
        selection.on('mouseenter.point', (d, i, g) => {
            let pointed = pointedElementSelector(g[i] as HTMLElement)
            pointed.attr('pointed', 'pointed');
            if (callback != null) callback(pointed);
        });
        selection.on('mouseleave.point', (d, i, g) => {
            pointedElementSelector(g[i] as HTMLElement).attr('pointed', null);
            if (callback != null) callback(d3.selectAll() as any);
        });
    }

    export function addPickBehavior(viewer: Types.Viewer, selection: Types.D3Selection,
        pickedElementSelector = (clicked: HTMLElement) => (d3.select(clicked) as any as Types.D3Selection),
        callback: ((sel: Types.D3Selection) => any) | null = null) {
        selection.on('click.pick', (d, i, g) => {
            let d3Elem = pickedElementSelector(g[i] as HTMLElement);
            d3Elem.attr('picked', 'pre-picked');
        });
        viewer.guiDiv
            .on('click.pick', () => {
                viewer.guiDiv.selectAll('[picked=picked]').attr('picked', null);
                let picked = viewer.guiDiv.selectAll('[picked=pre-picked]');
                picked.attr('picked', 'picked');
                if (callback != null) callback(picked);
            });
    }

    export function addMouseHoldBehavior(selection: Types.D3Selection, onDown: () => any, onHold: () => any, onUp: () => any) {
        selection.on('mousedown', async () => {
            if (d3.event.which == 1 || d3.event.which == undefined) { // d3.event.which: 1=left, 2=middle, 3=right mouse button
                let thisClickId = Math.random().toString(36).slice(2);
                onDown();
                selection.attr('pressed', thisClickId);
                await sleep(Constants.MOUSE_HOLD_BEHAVIOR_INITIAL_SLEEP_TIME);
                while (selection.attr('pressed') == thisClickId) {
                    onHold();
                    await sleep(Constants.MOUSE_HOLD_BEHAVIOR_STEP_SLEEP_TIME);
                }
            }
        });
        selection.on('mouseup', async () => {
            selection.attr('pressed', null);
            onUp();
        });
        selection.on('mouseleave', async () => {
            selection.attr('pressed', null);
            onUp();
        });
    }
    function sleep(ms: number) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    export function recolor(viewer: Types.Viewer, transition = true): void {
        let duration = transition ? Constants.TRANSITION_DURATION : 0;
        viewer.canvas
            .select('g.nodes')
            .selectAll('g.node')
            .selectAll('rect,polygon')
            .transition().duration(duration)
            .style('fill', n => (n as Dag.Node).visual.fill)
            .style('stroke', n => (n as Dag.Node).visual.stroke);
        let betaArcs = viewer.canvas
            .select('g.beta-connectivity')
            .selectAll('g.ladder')
            .selectAll('path.vis');
        if (viewer.settings.colorMethod != Enums.ColorMethod.Stdev) {
            betaArcs.style('stroke', ladder => viewer.data.nodes[(ladder as Dag.Edge)[0]].visual.stroke);
        } else {
            let arcColor = Colors.NEUTRAL_DARK.hex();
            betaArcs.style('stroke', arcColor);
        }
        show3DVariabilityLegend(viewer, viewer.settings.colorMethod == Enums.ColorMethod.Stdev, transition);
    }


    export function redraw(viewer: Types.Viewer, transition = true): void {
        let duration = transition ? Constants.TRANSITION_DURATION : 0;
        let d3nodes = viewer.canvas
            .select('g.nodes')
            .selectAll('g.node')
            .filter(n => (n as Dag.Node).active);
        d3nodes.select('rect')
            .transition().duration(duration)
            .attrs(n => Geometry.rectToScreen(viewer.visWorld, viewer.screen, (n as Dag.Node).visual.rect));
        d3nodes.select('polygon')
            .transition().duration(duration)
            .attr('points', n => Geometry.symCdfPolygonPoints(Geometry.rectToScreen(viewer.visWorld, viewer.screen, (n as Dag.Node).visual.rect), (n as Dag.Node).cdf));
        d3nodes.select('text')
            .transition().duration(duration)
            .attrs(n => {
                let { x, y, height, width } = Geometry.rectToScreen(viewer.visWorld, viewer.screen, (n as Dag.Node).visual.rect);
                return { x: x + width / 2, y: y + height + Constants.HANGING_TEXT_OFFSET }
            })
            .style('opacity', n => nodeBigEnoughForLabel(viewer, n as Dag.Node) ? 1 : 0);
        viewer.canvas
            .select('g.edges')
            .selectAll('line')
            .transition().duration(duration)
            .attrs(line => Geometry.lineToScreen(viewer.visWorld, viewer.screen, line as Geometry.Line));
        let arcMaxMinor = Constants.ARC_MAX_MINOR / viewer.zoom.yZoomout;
        // let arcMaxDeviation = Constants.ARC_MAX_DEVIATION / viewer.zoom.yZoomout;
        // let arcSmartDeviationParam = viewer.world.width * Constants.ARC_SMART_DEVIATION_PARAM_WRT_WORLD_WIDTH / viewer.zoom.xZoomout;
        let dag = viewer.data;
        dag.beta_connectivity.forEach(edge => edge[3] = dag.nodes[edge[0]].active && dag.nodes[edge[1]].active ? 1 : 0);
        viewer.canvas
            .select('g.beta-connectivity')
            .selectAll('path')
            .transition().duration(duration)
            .attr('d', edge => calculateArcPath(viewer, edge as Dag.Edge));
    }

    function calculateArcPath(viewer: Types.Viewer, edge: Dag.Edge): string {
        if ((edge as Number[])[3] == 0) return '';
        let [u, v, orientation] = edge as number[];
        let n1 = viewer.data.nodes[u].visual.rect;
        let n2 = viewer.data.nodes[v].visual.rect;
        let endpoints = Geometry.lineToScreen(viewer.visWorld, viewer.screen, { x1: n1.x + 0.5 * n1.width, y1: n1.y + 0.5 * n1.height, x2: n2.x + 0.5 * n2.width, y2: n2.y + 0.5 * n2.height });
        let x2yZoomRatio = viewer.zoom.yZoomout / viewer.zoom.xZoomout;
        let arcMaxMinor = Constants.ARC_MAX_MINOR / viewer.zoom.yZoomout;
        // return Geometry.arcPathD_circle(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
        return Geometry.arcPathD_ellipse(endpoints, viewer.world.width / viewer.zoom.xZoomout, arcMaxMinor, orientation == 1, x2yZoomRatio);
        // return Geometry.arcPathD_bezier(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
    }

    export function nodeBigEnoughForLabel(viewer: Types.Viewer, node: Dag.Node) {
        return Geometry.rectToScreen(viewer.visWorld, viewer.screen, node.visual.rect).width >= Constants.MINIMAL_WIDTH_FOR_SSE_LABEL;
    }

    function gradientBar(canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>, rect: Geometry.Rectangle): Types.D3Selection {
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
                }
            });
        bar.append('rect')
            .attrs(i => { return { class: 'border', ...rect, fill: 'none' } });
        return bar as unknown as Types.D3Selection;
    }

    function gradientBarExp(canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>, rect: Geometry.Rectangle,
        maxValue: number, middle: number, ticksDistance: number, legend = ''): Types.D3Selection {
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
            .attrs(i => { return { x: i * step, y: 0, width: step + 1, height: rect.height, stroke: 'none', fill: Colors.byExpHeatmap(i / n * maxValue, middle).hex() } });
        barStroke.append('rect')
            .attrs(i => { return { x: 0, y: 0, width: rect.width, height: rect.height, fill: 'none' } });
        let barTicks = bar.append('g').attr('class', 'heatmap-bar-ticks');
        let barTickTexts = barTicks.selectAll('text')
            .data(d3.range(Math.floor(maxValue / ticksDistance) + 1))
            .enter()
            .append('text')
            .attr('x', i => i * ticksDistance / maxValue * rect.width)
            .attr('y', rect.height + Constants.HANGING_TEXT_OFFSET)
            .text(i => i * ticksDistance);
        return bar as unknown as Types.D3Selection;
    }

    function moveGradientBarExp(bar: Types.D3Selection, newX: number, newY: number, ticksAbove: boolean): void {
        bar.attr('transform', `translate(${newX},${newY})`);
        let barTickTexts = bar.select('g.heatmap-bar-ticks').selectAll('text');
        let barHeight = bar.select('g.heatmap-bar-stroke rect').attr('height');
        if (ticksAbove) {
            barTickTexts.attr('y', -Constants.HANGING_TEXT_OFFSET);
            barTickTexts.style('dominant-baseline', 'alphabetic');  // For some reason 'alphabetic' means 'bottom'
        } else {
            barTickTexts.attr('y', barHeight + Constants.HANGING_TEXT_OFFSET);
            barTickTexts.style('dominant-baseline', 'hanging');
        }
    }

    function show3DVariabilityLegend(viewer: Types.Viewer, on: boolean, transition: boolean = true): void {
        fadeOutRemove(viewer.canvas.selectAll('g.heatmap-bar'));
        let BAR_WIDTH = 200;
        let BAR_HEIGHT = 20;
        let BAR_HMARGIN = 15;
        let BAR_VMARGIN = 5;
        if (on) {
            let bar = gradientBarExp(viewer.canvas, { x: viewer.screen.width - BAR_WIDTH - BAR_HMARGIN, y: BAR_VMARGIN, width: BAR_WIDTH, height: BAR_HEIGHT },
                15, 5, 5, '3D variability [\u212B]');
            let controlsRight = (viewer.mainDiv.select('div.control-panel#main-panel').node() as Element | null)?.getBoundingClientRect()?.right;
            let barLeft = (bar.node() as Element | null)?.getBoundingClientRect()?.left;
            if (controlsRight !== undefined && barLeft !== undefined && controlsRight > barLeft) {
                moveGradientBarExp(bar, viewer.screen.width - BAR_WIDTH - BAR_HMARGIN, viewer.screen.height - BAR_VMARGIN - BAR_HEIGHT, true);
            }
            if (transition) {
                fadeIn(bar);
            }
        }
    }

    export function showBetaConnectivity(viewer: Types.Viewer, on: boolean, transition = true): void {
        viewer.settings.betaConnectivityVisibility = on;
        let oldBetaConnectivityVis = viewer.canvas.selectAll('g.beta-connectivity');
        if (oldBetaConnectivityVis.size() > 0 && !on) {
            if (transition) fadeOutRemove(oldBetaConnectivityVis);
            else oldBetaConnectivityVis.remove();
        } else if (oldBetaConnectivityVis.size() == 0 && on) {
            let dag = viewer.data;
            dag.beta_connectivity.forEach(edge => edge[3] = dag.nodes[edge[0]].active && dag.nodes[edge[1]].active ? 1 : 0);
            let betaConnectivityVis = viewer.canvas
                .append('g').attr('class', 'beta-connectivity');
            // let betaConnectivityPaths = betaConnectivityVis.selectAll()
            //     .data(dag.beta_connectivity)
            //     .enter()
            //     .append('path')
            //     .style('stroke', ladder => dag.nodes[ladder[0]].visual.stroke);
            let betaConnectivityLadders = betaConnectivityVis.selectAll()
                .data(dag.beta_connectivity)
                .enter()
                .append('g').attr('class', 'ladder');
            viewer.ladderMap = new Types.TupleMap<string, SVGElement>();
            betaConnectivityLadders.each((d, i, elems) => {
                const strand1 = viewer.data.nodes[d[0]].label;
                const strand2 = viewer.data.nodes[d[1]].label;
                const dir = d[2] > 0 ? 'parallel' : 'antiparallel';
                viewer.ladderMap.set([strand1, strand2, dir], elems[i]);
            });
            // console.log('ladderMap', viewer.ladderMap.entries());
            let betaPaths = betaConnectivityLadders
                .append('path').attr('class', 'vis')
                .style('stroke', ladder => dag.nodes[ladder[0]].visual.stroke);
            let betaGhostPaths = betaConnectivityLadders
                .append('path').attr('class', 'ghost');
            // addPointBehavior(viewer, betaConnectivityPaths as any);
            addPointBehavior(viewer, betaGhostPaths as any, path => selectLadderFromGhostPath(viewer, path as any, true),
                nodes => Drawing.dispatchMixedEvent(viewer, Constants.EVENT_TYPE_HOVER, nodes.data()));
            redraw(viewer, false);
            if (transition) {
                fadeIn(betaConnectivityVis as unknown as Types.D3Selection);
            }
        }
    }

    function selectLadderFromGhostPath(viewer: Types.Viewer, path: SVGElement, includeStrands: boolean): Types.D3Selection {
        let ladder = d3.select(path.parentElement);
        if (!includeStrands) {
            return ladder as any;
        }
        const edge = ladder.data()[0] as Dag.Edge;
        let nodes = viewer.canvas.selectAll('g.node').nodes();
        const m = d3.selectAll([ladder.node(), nodes[edge[0]], nodes[edge[1]]]);
        return m;
    }

    export function selectNodeFromShape(viewer: Types.Viewer, shape: SVGElement, includeLadders: boolean): Types.D3Selection {
        let gnode = d3.select(shape.parentElement);
        if (!includeLadders) {
            return gnode as any;
        }
        let result: d3.BaseType[] = [gnode.node()];
        const node = gnode.data()[0] as Dag.Node;
        if (node.ladders != undefined && node.ladders.length>0){
            let gladders = viewer.canvas.selectAll('g.ladder').nodes();
            for (const i of node.ladders) result.push(gladders[i]);
        }
        return d3.selectAll(result);
    }

    function concatSelections(...selections: Types.D3Selection[]): Types.D3Selection {
        let nodes = [];
        for (const selection of selections) nodes.push(...selection.nodes());
        return d3.selectAll(nodes);
    }

    export function fadeOutRemove(selection: Types.D3Selection, delay = 0): Types.D3Transition {
        return selection.transition().delay(delay).duration(Constants.TRANSITION_DURATION).style('opacity', 0).remove();
    }

    export function fadeIn(selection: Types.D3Selection, delay = 0): Types.D3Transition {
        if (selection.size() == 0) {
            return selection.transition();
        }
        let op = selection.style('opacity');
        return selection
            .style('opacity', 0)
            .transition().delay(delay).duration(Constants.TRANSITION_DURATION)
            .style('opacity', op);
    }

    function constrainVisWorldXY(viewer: Types.Viewer) {
        let { x: wx, y: wy, width: ww, height: wh } = viewer.world;
        let { x: vx, y: vy, width: vw, height: vh } = viewer.visWorld;
        if (vw < ww * Constants.MAX_X_ZOOMOUT) {
            viewer.visWorld.x = Geometry.constrain(
                vx,
                wx - Constants.MAX_EMPTY_X_MARGIN * vw,
                wx + ww + Constants.MAX_EMPTY_X_MARGIN * vw - vw);
        } else {
            viewer.visWorld.x = wx + 0.5 * ww - 0.5 * vw;
        }
        if (vh < wh * Constants.MAX_Y_ZOOMOUT) {
            viewer.visWorld.y = Geometry.constrain(
                vy,
                wy - Constants.MAX_EMPTY_Y_MARGIN * vh,
                wy + wh + Constants.MAX_EMPTY_Y_MARGIN * vh - vh);
        } else {
            viewer.visWorld.y = wy + 0.5 * wh - 0.5 * vh;
        }
    }

    export function dispatchMixedEvent(viewer: Types.Viewer, eventType: string, targets: (Dag.Node | Dag.Edge)[]): void {
        if (!viewer.settings.dispatchEvents) return;
        let sses = [];
        let edges = [];
        for (const target of targets) {
            if ((target as any).label != undefined) {  // is SSE
                const sse = target as Dag.Node;
                sses.push({
                    label: sse.label,
                    type: sse.type,
                    sheetId: sse.type.toLowerCase() == 'e' ? sse.sheet_id : null,
                });
            } else {  // is ladder
                const edge = target as Dag.Edge;
                edges.push([
                    viewer.data.nodes[edge[0]].label,
                    viewer.data.nodes[edge[1]].label,
                    edge[2] > 0 ? 'parallel' : 'antiparallel',
                ]);
            }
        }
        let eventDetail = {
            sourceType: viewer.d3viewer.node()?.tagName.toLowerCase(),
            sourceId: viewer.id,
            sourceInternalId: viewer.internalId,
            eventType: eventType,
            sses: sses,
            ladders: edges,
        };
        viewer.mainDiv.dispatch(Constants.EVENT_PREFIX + eventType, { detail: eventDetail, bubbles: true } as any);
    }

    export function handleEvent(viewer: Types.Viewer, event: CustomEvent): void {
        // console.log('Inbound event', event.type, event);
        const detail = event.detail;
        if (detail == null || detail == undefined) {
            console.error(`Event ${event.type}: event.detail must be an object.`);
            return;
        }
        if (detail.sourceType == viewer.d3viewer.node()?.tagName.toLowerCase() && detail.sourceInternalId == viewer.internalId) {
            // console.log('Ignoring self', viewer.uniqueId);
            return;
        }
        const sses = detail.sses ?? [];
        const ladders = detail.ladders ?? [];
        const PDB_OVERPROT_DO_SELECT = Constants.EVENT_PREFIX + Constants.EVENT_TYPE_DO_SELECT;
        const PDB_OVERPROT_DO_HOVER = Constants.EVENT_PREFIX + Constants.EVENT_TYPE_DO_HOVER;
        let attribute: string;
        switch (event.type) {
            case PDB_OVERPROT_DO_SELECT:  // PDB.overprot.do.select
                attribute = 'picked';
                break;
            case PDB_OVERPROT_DO_HOVER:  // PDB.overprot.do.hover
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
            } else {
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
            const g = viewer.ladderMap.get([u, v, dir]) ?? viewer.ladderMap.get([v, u, dir]);
            if (g != undefined) {
                d3.select(g).attr(attribute, attribute);
            } else {
                console.warn(`Event ${event.type}: Beta-ladder ${ladder} is not present.`);
            }
        }

        if (event.type == PDB_OVERPROT_DO_SELECT) {
            unpinAllTooltips(viewer);
        }
    }

}