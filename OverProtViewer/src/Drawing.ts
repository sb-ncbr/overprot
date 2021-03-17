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
            x: centerX - 0.5*visWidth,
            y: centerY - 0.5*visHeight,
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
        if (!hasPinnedTooltip){
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


    export function setTooltips(viewer: Types.Viewer, selection: Types.D3Selection, htmlContents: (string|null)[]|null, pinnable = false, delay = false) {
        if (htmlContents === null){
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

    export function addPointBehavior(selection: Types.D3Selection, pointedElementSelector = (pointed: HTMLElement) => (d3.select(pointed) as any as Types.D3Selection)){
        selection.on('mouseenter.point', (d, i, g) => {
            pointedElementSelector(g[i] as HTMLElement).attr('pointed', 'true');
        });
        selection.on('mouseleave.point', (d, i, g) => {
            pointedElementSelector(g[i] as HTMLElement).attr('pointed', null);
        });
    }

    export function addPickBehavior(viewer: Types.Viewer, selection: Types.D3Selection, pickedElementSelector = (clicked: HTMLElement) => (d3.select(clicked) as any as Types.D3Selection)){
        selection.on('click.pick', (d, i, g) => {
            pickedElementSelector(g[i] as HTMLElement).attr('picked', 'pre-picked');
        });
        viewer.guiDiv
            .on('click.pick', () => {
                viewer.guiDiv.selectAll('[picked=picked]').attr('picked', null);
                viewer.guiDiv.selectAll('[picked=pre-picked]').attr('picked', 'picked');
            });
    }

    export function addMouseHoldBehavior(selection: Types.D3Selection, onDown: ()=>any, onHold: ()=>any, onUp: ()=>any) {
        selection.on('mousedown', async () => {
            if (d3.event.which == 1 || d3.event.which == undefined) { // d3.event.which: 1=left, 2=middle, 3=right mouse button
                // console.log('mousedown', d3.event.which, d3.event.button, d3.event.buttons);
                let thisClickId = Math.random().toString(36).slice(2);
                onDown();
                selection.attr('pressed', thisClickId);
                await sleep(Constants.MOUSE_HOLD_BEHAVIOR_INITIAL_SLEEP_TIME);            
                while (selection.attr('pressed') == thisClickId) {
                    onHold();
                    await sleep(Constants.MOUSE_HOLD_BEHAVIOR_STEP_SLEEP_TIME);
                    // console.log('still down?', selection.attr('pressed'));
                }
            }
        });
        selection.on('mouseup', async () => {
            selection.attr('pressed', null);
            // console.log('mouseup');
            onUp();
        });
        selection.on('mouseleave', async () => {
            selection.attr('pressed', null);
            // console.log('mouseup');
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
            .attr('d', edge => {
                if ((edge as Number[])[3] == 0) return '';
                let [u, v, orientation] = edge as number[];
                let n1 = viewer.data.nodes[u].visual.rect;
                let n2 = viewer.data.nodes[v].visual.rect;
                let endpoints = Geometry.lineToScreen(viewer.visWorld, viewer.screen, { x1: n1.x + 0.5 * n1.width, y1: n1.y + 0.5 * n1.height, x2: n2.x + 0.5 * n2.width, y2: n2.y + 0.5 * n2.height });
                let x2yZoomRatio = viewer.zoom.yZoomout / viewer.zoom.xZoomout;
                // return Geometry.arcPathD_circle(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
                return Geometry.arcPathD_ellipse(endpoints, viewer.world.width / viewer.zoom.xZoomout, arcMaxMinor, orientation == 1, x2yZoomRatio);
                // return Geometry.arcPathD_bezier(endpoints, arcMaxDeviation, arcSmartDeviationParam, orientation == 1, x2yZoomRatio);
            });
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
        .attr('x', i => i*ticksDistance / maxValue * rect.width)
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
            let controlsRight = (viewer.mainDiv.select('div.control-panel#main-panel').node() as Element|null)?.getBoundingClientRect()?.right;
            let barLeft = (bar.node() as Element|null)?.getBoundingClientRect()?.left;
            if (controlsRight !== undefined && barLeft !== undefined && controlsRight > barLeft){
                moveGradientBarExp(bar, viewer.screen.width - BAR_WIDTH - BAR_HMARGIN, viewer.screen.height - BAR_VMARGIN - BAR_HEIGHT, true);
            }
            if (transition){
                fadeIn(bar);
            }
        }
    }

    export function showBetaConnectivity(viewer: Types.Viewer, on: boolean, transition = true): void {
        viewer.settings.betaConnectivityVisibility = on;
        let oldBetaConnectivityVis = viewer.canvas.selectAll('g.beta-connectivity');
        if (oldBetaConnectivityVis.size() > 0 && !on) {
            console.log('Hiding beta-connectivity.');
            if (transition) fadeOutRemove(oldBetaConnectivityVis);
            else oldBetaConnectivityVis.remove();
        } else if (oldBetaConnectivityVis.size() == 0 && on) {
            console.log('Showing beta-connectivity.');
            let dag = viewer.data;
            dag.beta_connectivity.forEach(edge => edge[3] = dag.nodes[edge[0]].active && dag.nodes[edge[1]].active ? 1 : 0);
            let betaConnectivityVis = viewer.canvas
                .append('g').attr('class', 'beta-connectivity');
            let betaConnectivityPaths = betaConnectivityVis.selectAll()
                .data(dag.beta_connectivity)
                .enter()
                .append('path')
                .style('stroke', ladder => dag.nodes[ladder[0]].visual.stroke);
            addPointBehavior(betaConnectivityPaths as any);
            redraw(viewer, false);
            if (transition) {
                fadeIn(betaConnectivityVis as unknown as Types.D3Selection);
            }
        }
    }

    export function fadeOutRemove(selection: Types.D3Selection, delay = 0): Types.D3Transition {
        return selection.transition().delay(delay).duration(Constants.TRANSITION_DURATION).style('opacity', 0).remove();
    }

    export function fadeIn(selection: Types.D3Selection, delay = 0): Types.D3Transition {
        if (selection.size() == 0){
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
            viewer.visWorld.x = wx + 0.5*ww - 0.5*vw;
        }
        if (vh < wh * Constants.MAX_Y_ZOOMOUT) {
            viewer.visWorld.y = Geometry.constrain(
                vy,
                wy - Constants.MAX_EMPTY_Y_MARGIN * vh,
                wy + wh + Constants.MAX_EMPTY_Y_MARGIN * vh - vh);
        } else {
            viewer.visWorld.y = wy + 0.5*wh  - 0.5*vh;
        }
    }

}