import * as d3 from 'd3';
import * as d3SelectionMulti from 'd3-selection-multi';

import { Constants } from './Constants';
import { Colors } from './Colors';
import { Types } from './Types';
import { Dag } from './Dag';
import { Graphs } from './Graphs';
import { Drawing } from './Drawing';
import { Geometry } from './Geometry';
import { Enums } from './Enums';
import { Controls } from './Controls';
import { tree, select, html } from 'd3';


export namespace OverProtViewerCore {

    export function initializeViewer(htmlElement: HTMLElement): void {
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
        let realSize = d3guiDiv.node()?.getBoundingClientRect();
        if (realSize !== undefined) {
            let { width, height } = realSize;
            settings.width = width;
            settings.height = height;
        }

        let d3canvasDiv = d3guiDiv.append('div').attr('class', 'canvas');
        let d3canvas = d3canvasDiv.append('svg').attr('class', 'canvas')
            .attrs({ width: settings.width, height: settings.height });

        let viewer: Types.Viewer = Types.newViewer(id, uniqueId, d3viewer, d3mainDiv, d3guiDiv, d3canvas, settings);
        
        // initializeExternalControls(viewer);
        initializeInternalControls(viewer);

        d3mainDiv.append('br');

        d3canvas.on('dblclick', () => Drawing.zoomAll(viewer));

        d3canvas.on('wheel', () => {
            if (d3.event.ctrlKey) {
                let ratio = d3.event.deltaY > 0 ? Constants.ZOOM_STEP_RATIO_MOUSE : 1 / Constants.ZOOM_STEP_RATIO_MOUSE;
                Drawing.zoomOut(viewer, ratio, d3.mouse(d3canvas.node() as d3.ContainerElement));
                d3.event.preventDefault();
            } else {
                let messageBackground = viewer.canvas.selectAll('text.temporary-central-message').data([1])
                    .enter().append('rect').attr('class', 'temporary-central-message-background')
                    .attrs({ x: 0, width: viewer.screen.width, y: 0, height: viewer.screen.height });
                let message = viewer.canvas.selectAll('text.temporary-central-message').data([1])
                    .enter().append('text').attr('class', 'temporary-central-message')
                    .attrs({ x: viewer.screen.width / 2, y: viewer.screen.height / 2 })
                    .text('Use Ctrl + scroll to zoom');
                Drawing.fadeIn(messageBackground as any);
                Drawing.fadeIn(message as any);
                Drawing.fadeOutRemove(messageBackground as any, Constants.TEMPORARY_CENTRAL_MESSAGE_TIMEOUT);
                Drawing.fadeOutRemove(message as any, Constants.TEMPORARY_CENTRAL_MESSAGE_TIMEOUT);
            }
        });

        let dragHandler = d3.drag()
            .on('drag', () => Drawing.shiftByMouse(viewer, -d3.event.dx, -d3.event.dy));
        dragHandler(d3canvas as unknown as d3.Selection<Element, unknown, any, any>);

        d3.select(window).on('resize.' + uniqueId, () => {
            let newSize = d3guiDiv.node()!.getBoundingClientRect();
            resizeVisualization(viewer, newSize.width, newSize.height);
        });

        if (settings.listenEvents){
            htmlElement.addEventListener(Constants.EVENT_PREFIX+Constants.EVENT_TYPE_DO_SELECT, (e) => Drawing.handleEvent(viewer, e as CustomEvent));  // does not work with d3 .on() because of special meaning of dots in event type
            htmlElement.addEventListener(Constants.EVENT_PREFIX+Constants.EVENT_TYPE_DO_HOVER, (e) => Drawing.handleEvent(viewer, e as CustomEvent));
        }
        viewer.canvas.append('text').attr('class', 'central-message')
            .attrs({ x: viewer.screen.width / 2, y: viewer.screen.height / 2 })
            .text('Loading...');

        if (settings.file != '') {
            let absoluteFilePath = new URL(settings.file, document.baseURI).toString();
            // console.log(`Fetching ${absoluteFilePath}`);
            fetch(absoluteFilePath)
                .catch(async error => {
                    console.error(`OverProt Viewer with id="${viewer.id}" failed to initialize because the request for`, absoluteFilePath, 
                        'failed (this might be caused by CORS policy). Original error:', error);
                    let data = Dag.newDagWithError(`Failed to fetch data from "${absoluteFilePath}"`);
                    setDataToViewer(viewer, data);
                })
                .then(async response => {
                    if (response && response.ok) {
                        let text = await response.text();
                        let data = Dag.dagFromJson(text);
                        setDataToViewer(viewer, data);
                    } else if (response && !response.ok){
                        console.error(`OverProt Viewer with id="${viewer.id}" failed to initialize because the request for`, absoluteFilePath, 
                            'returned status code', response.status, response.statusText);
                        let data = Dag.newDagWithError(`Failed to fetch data from "${absoluteFilePath}"`);
                        setDataToViewer(viewer, data);
                    }
                })
                .catch(async error => {
                    console.error(`OverProt Viewer with id="${viewer.id}" failed to initialize because of internal error.`);
                    let data = Dag.newDagWithError(`Internal error`);
                    setDataToViewer(viewer, data);
                    throw error;
                });
        } else {
            let data = Dag.newDagWithError('No file specified');
            setDataToViewer(viewer, data);
        }

    }
    function setkv(obj: any, key: string, value: any): void{
        obj[key] = value;
    }

    function initializeExternalControls(viewer: Types.Viewer): void {
        let d3controlsDiv = viewer.mainDiv.append('div').attr('id', 'controls');

        // ZOOM CONTROLS
        let d3zoomDiv = d3controlsDiv.append('div').attr('id', 'zoom');
        let d3zoomSpan = d3zoomDiv.append('span') as unknown as Types.D3Selection;
        d3zoomSpan.append('label').text('Zoom:');
        d3zoomSpan.append('button').text('+').on('click', () => Drawing.zoomIn(viewer));
        d3zoomSpan.append('button').text('-').on('click', () => Drawing.zoomOut(viewer));
        d3zoomSpan.append('button').text('Reset').on('click', () => Drawing.zoomAll(viewer));
        Drawing.setTooltips(viewer, d3zoomSpan, [`<div class="control-tooltip">Alternatively, use mouse wheel to zoom, mouse drag to shift, double-click to reset.</div>`], false, true);
        d3zoomDiv.append('br');

        // COLOR CONTROLS
        let d3colorRadioDiv = d3controlsDiv.append('div').attr('class', 'radio-group').attr('id', 'color');
        d3colorRadioDiv.append('label').text('Color:');
        let radioGroupName = `color-radio-${viewer.uniqueId}`;
        let options = [
            { label: 'Uniform', value: 'uniform', method: Enums.ColorMethod.Uniform, tip: 'Show all SSEs in the same color.' },
            { label: 'Type', value: 'type', method: Enums.ColorMethod.Type, tip: 'Show &beta;-strands in blue, helices in gray.' },
            { label: 'Sheet', value: 'sheet', method: Enums.ColorMethod.Sheet, tip: 'Assign the same color to &beta;-strands from the same &beta;-sheet, <br>show helices in gray.' },
            { label: '3D Variability', value: 'variability', method: Enums.ColorMethod.Stdev, tip: '<strong>3D variability</strong> measures the standard deviation of the SSE end point coordinates.<br>Low values (dark) indicate conserved SSE position, <br>high values (bright) indicate variable SSE position.' }
        ];
        for (const option of options) {
            let radioId = `radio-${option.value}-${viewer.uniqueId}`;
            let radioSpan = d3colorRadioDiv.append('span') as unknown as Types.D3Selection;
            radioSpan.append('input').attrs({ type: 'radio', name: radioGroupName, id: radioId })
                .property('checked', viewer.settings.colorMethod == option.method)
                .on('change', () => applyColors(viewer, option.method));
            radioSpan.append('label').attr('for', radioId).text(option.label);
            Drawing.setTooltips(viewer, radioSpan, [`<div class="control-tooltip">${option.tip}</div>`], false, true);
        }
        d3colorRadioDiv.append('br');

        // SHAPE CONTROLS
        let d3shapeRadioDiv = d3controlsDiv.append('div').attr('class', 'radio-group').attr('id', 'shape');
        d3shapeRadioDiv.append('label').text('Shape:');
        let shapeRadioGroupName = `shape-radio-${viewer.uniqueId}`;
        let shapeOptions = [
            { label: 'Rectangle', value: 'rectangle', method: Enums.ShapeMethod.Rectangle, tip: 'Show SSEs as rectangles. <br>Height of the rectangle indicates <strong>occurrence</strong> (what percentage of structures contain this SSE), <br>width indicates <strong>average length</strong> (number of residues).' },
            { label: 'Symmetric CDF', value: 'symcdf', method: Enums.ShapeMethod.SymCdf, tip: '<strong>Cumulative distribution function</strong> describes the statistical distribution of the SSE length. <br>The widest part of the shape corresponds to maximum length, the narrowest to minimum length, <br> the height corresponds to occurrence. <br>(The shape consists of four symmetrical copies of the CDF, the bottom right quarter is the classical CDF.)' },
        ];
        for (const option of shapeOptions) {
            let radioId = `radio-${option.value}-${viewer.uniqueId}`;
            let radioSpan = d3shapeRadioDiv.append('span') as unknown as Types.D3Selection;
            radioSpan.append('input').attrs({ type: 'radio', name: shapeRadioGroupName, id: radioId })
                .property('checked', viewer.settings.shapeMethod == option.method)
                .on('change', () => applyShapes(viewer, option.method));
            radioSpan.append('label').attr('for', radioId).text(option.label);
            Drawing.setTooltips(viewer, radioSpan, [`<div class="control-tooltip">${option.tip}</div>`], false, true);
        }
        d3shapeRadioDiv.append('br');

        // BETA-CONNECTIVITY CONTROLS
        let d3betaConnectivityDiv = d3controlsDiv.append('div').attr('id', 'beta-connectivity');
        let d3betaConnectivitySpan = d3betaConnectivityDiv.append('span') as unknown as Types.D3Selection;
        let checkboxId = `checkbox-beta-connectivity-${viewer.uniqueId}`;
        d3betaConnectivitySpan.append('label').attr('for', checkboxId).text('Beta-connectivity:');
        d3betaConnectivitySpan.append('input').attr('type', 'checkbox').attr('id', checkboxId)
            .property('checked', viewer.settings.betaConnectivityVisibility)
            .on('change', () => Drawing.showBetaConnectivity(viewer, d3.event.target.checked, true)) as unknown as Types.D3Selection;
        let tip = ['<div class="control-tooltip">',
            '<strong>Beta-connectivity</strong> shows how &beta;-strands are connected to each other in &beta;-sheets.<br>',
            'Upper arcs indicate antiparallel connections,<br>',
            'lower arcs indicate parallel connections.',
            '</div>'
        ].join('');
        Drawing.setTooltips(viewer, d3betaConnectivitySpan, [tip], false, true);
        d3betaConnectivityDiv.append('br');

        // OCCURRENCE THRESHOLD CONTROLS
        let d3sliderDiv = d3controlsDiv.append('div').attr('class', 'ext-slider').attr('id', 'occurrence-threshold') as unknown as Types.D3Selection;
        let sliderId = 'occurrence-threshold-slider-' + viewer.uniqueId;
        d3sliderDiv.append('label').attr('for', sliderId).text('Occurrence threshold: ')
            .append('span').attr('class', 'slider-indicator').attr('id', 'occurrence-threshold-indicator').text(viewer.settings.occurrenceThreshold * 100 + '%');
        d3sliderDiv.append('br');
        let d3slider = d3sliderDiv.append('input').attr('class', 'slider')
            .attrs({ type: 'range', id: sliderId, min: 0, step: 1, max: 100, value: viewer.settings.occurrenceThreshold * 100 });
        d3slider.on('input', () => d3sliderDiv.select('#occurrence-threshold-indicator').text(d3.event.target.valueAsNumber + '%'));  // While holding mouse button
        d3slider.on('change', () => applyFiltering(viewer, d3.event.target.valueAsNumber / 100));  // When releasing mouse button
        Drawing.setTooltips(viewer, d3sliderDiv, ['<div class="control-tooltip">Hide SSEs with occurrence lower than the specified threshold.</div>'], false, true);
    }


    function initializeInternalControls(viewer: Types.Viewer): void {
        let controlsDiv = viewer.guiDiv.append('div').attr('class', 'internal-controls');

        let colorOptions: Controls.NameValueTooltip<Enums.ColorMethod>[] = [
            ['Uniform', Enums.ColorMethod.Uniform, 'Show all SSEs in the same color.'],
            ['Type', Enums.ColorMethod.Type, 'Show &beta;-strands in blue, helices in gray.'],
            ['Sheet', Enums.ColorMethod.Sheet, 'Assign the same color to &beta;-strands from the same &beta;-sheet, <br>show helices in gray.'],
            ['Variability', Enums.ColorMethod.Stdev, '<strong>3D variability</strong> measures the standard deviation of the SSE end point coordinates.<br>Low values (dark) indicate conserved SSE position, <br>high values (bright) indicate variable SSE position.']
        ];
        let shapeOptions: Controls.NameValueTooltip<Enums.ShapeMethod>[] = [
            ['Rectangle', Enums.ShapeMethod.Rectangle, 'Show SSEs as rectangles. <br>Height of the rectangle indicates <strong>occurrence</strong> (what percentage of structures contain this SSE), <br>width indicates <strong>average length</strong> (number of residues).'],
            ['SymCDF', Enums.ShapeMethod.SymCdf, '<strong>Cumulative distribution function</strong> describes the statistical distribution of the SSE length. <br>The widest part of the shape corresponds to maximum length, the narrowest to minimum length, <br> the height corresponds to occurrence. <br>(The SymCDF shape consists of four symmetrical copies of the CDF, the bottom right quarter is the classical CDF.)'],
        ];
        let connectivityOptions: Controls.NameValueTooltip<boolean>[] = [
            ['On', true, null],
            ['Off', false, null],
        ];

        let controlPanel = Controls.newControlPanel(viewer, 'main-panel', null);

        let zoomInButton = Controls.newButton(viewer, 'zoom-in', '+', true, () => Drawing.zoomIn(viewer), 'Zoom in. <br>Tip: Use Ctrl + mouse wheel to zoom.');
        Controls.addToControlPanel(controlPanel, zoomInButton);

        let zoomOutButton = Controls.newButton(viewer, 'zoom-out', '-', true, () => Drawing.zoomOut(viewer), 'Zoom out. <br>Tip: Use Ctrl + mouse wheel to zoom.');
        Controls.addToControlPanel(controlPanel, zoomOutButton);

        let zoomResetButoon = Controls.newButton(viewer, 'zoom-reset', Constants.RESET_SYMBOL, true, () => Drawing.zoomAll(viewer), 'Reset zoom. <br>Tip: Use double click to reset zoom.');
        Controls.addToControlPanel(controlPanel, zoomResetButoon);

        let colorMethodDropdown = Controls.newDropdownList(viewer, 'color-method', 'Color', colorOptions, viewer.settings.colorMethod,
            method => applyColors(viewer, method),
            'Choose coloring method.', true, false);
        Controls.addToControlPanel(controlPanel, colorMethodDropdown);

        let shapeMethodDropdown = Controls.newDropdownList(viewer, 'shape-method', 'Shape', shapeOptions, viewer.settings.shapeMethod,
            method => applyShapes(viewer, method),
            'Choose shape method.', true, false);
        Controls.addToControlPanel(controlPanel, shapeMethodDropdown);

        let betaConnectivityDropdown = Controls.newDropdownList(viewer, 'beta-connectivity', 'Beta-connectivity', connectivityOptions, viewer.settings.betaConnectivityVisibility,
            method => Drawing.showBetaConnectivity(viewer, method, true),
            '<strong>Beta-connectivity</strong> shows how &beta;-strands are connected to each other in &beta;-sheets.<br>Upper arcs indicate antiparallel connections,<br>lower arcs indicate parallel connections.',
            true, false);
        Controls.addToControlPanel(controlPanel, betaConnectivityDropdown);

        let occurrenceThresholdSlider = Controls.newPopupSlider(viewer, 'occurrence-threshold', 'Occurrence threshold: ', '%', 0, 100, 1, viewer.settings.occurrenceThreshold * 100,
            '0%', '100%',
            val => { },
            val => applyFiltering(viewer, val / 100),
            'Hide SSEs with occurrence lower than the specified threshold.');
        Controls.addToControlPanel(controlPanel, occurrenceThresholdSlider);

        // let occurrenceThresholdSlider2 = Controls.newSlider(viewer, 'occurrence-threshold', 0, 100, 1, viewer.settings.occurrenceThreshold*100, 
        //     '0%', '100%',
        //     val => {},
        //     val => applyFiltering(viewer, val / 100),
        //     'Hide SSEs with occurrence lower than the specified threshold.');
        // Controls.addToControlPanel(controlPanel, occurrenceThresholdSlider2);

        // let listbox = Controls2.newListbox(viewer, 'listbox', colorOptions, viewer.settings.colorMethod, v => {console.log(v);}, null);
        // Controls2.addToControlPanel(panelito, listbox);

        controlPanel.base.show(controlsDiv as any);

    }

    function setDataToViewer(viewer: Types.Viewer, data: Dag.Dag): void {
        viewer.data = data;
        Dag.filterDagAndAddLevels(viewer.data, viewer.settings.occurrenceThreshold);
        refreshVisualization(viewer);
    }

    function refreshVisualization(viewer: Types.Viewer): void {
        viewer.canvas.selectAll('*').remove();

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

        if (viewer.settings.layoutMethod == Enums.LayoutMethod.Old) {
            let nNodes = dag.nodes.length;
            let nLevels = dag.levels.length;
            let starts = new Array<number>(nNodes);
            let ends = new Array<number>(nNodes);
            let floors = new Array<number>(nNodes).fill(0);
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
                } else { // put the inactive nodes somewhere, where they don't cause problems
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
        } else if (viewer.settings.layoutMethod == Enums.LayoutMethod.New) {
            let graph = Graphs.newDagFromPrecedence(dag.levels, dag.precedence as any);
            let vertexSizes = new Map();
            for (const v of graph.vertices) vertexSizes.set(v, { width: reprLengths[v], height: reprHeights[v] });
            const [box, positions] = Graphs.embedDag(graph, vertexSizes, { x: Constants.GAP_LENGTH, y: Constants.FLOOR_HEIGHT - Constants.OCCURRENCE_SCALE },
                Constants.LEFT_MARGIN, Constants.RIGHT_MARGIN, Constants.TOP_MARGIN, Constants.BOTTOM_MARGIN);
            viewer.world = { x: -box.left, y: -box.top, width: box.right + box.left, height: box.bottom + box.top };
            for (let iNode = 0; iNode < dag.nodes.length; iNode++) {
                dag.nodes[iNode].visual = Dag.newNodeVisual();
                if (dag.nodes[iNode].active) {
                    const { x, y } = positions.get(iNode)!;
                    const { width, height } = vertexSizes.get(iNode)!;
                    dag.nodes[iNode].visual.rect = { x: x - width / 2, y: y - height / 2, width: width, height: height };
                } else { // put the inactive nodes somewhere, where they don't cause problems
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

        let nodeMap = new Map<string, SVGElement>();
        let d3nodes = viewer.canvas
            .append('g').attr('class', 'nodes')
            .selectAll('g.node')
            .data(dag.nodes)
            .enter()
            .append('g').attr('class', 'node')
            .attr('opacity', n => n.active ? 1 : 0)
            .each((d, i, nodes) => nodeMap.set(d.label, nodes[i]));
        viewer.nodeMap = nodeMap;
        console.log('viewer.nodeMap:', viewer.nodeMap);
        let d3nodeShapes: Types.D3Selection;
        if (viewer.settings.shapeMethod == Enums.ShapeMethod.SymCdf) {
            d3nodeShapes = d3nodes
                .append('polygon') as any;
        } else {
            d3nodeShapes = d3nodes
                .append('rect') as any;
        }
        let d3activeShapes = d3nodeShapes.filter(n => n.active);
        // Drawing.addPointBehavior(viewer, d3nodeShapes, shape => d3.select(shape.parentElement) as any);
        Drawing.addPointBehavior(viewer, d3nodeShapes, shape => d3.select(shape.parentElement) as any,
            nodes => Drawing.dispatchSseEvent(viewer, Constants.EVENT_TYPE_HOVER, nodes.data()));
        // Drawing.addPickBehavior(viewer, d3nodeShapes, shape => d3.select(shape.parentElement as any));
        Drawing.addPickBehavior(viewer, d3nodeShapes, shape => d3.select(shape.parentElement) as any, 
            nodes => Drawing.dispatchSseEvent(viewer, Constants.EVENT_TYPE_SELECT, nodes.data()));
        Drawing.setTooltips(viewer, d3nodeShapes as any, d3nodes.data().map(createNodeTooltip), true, false);

        d3nodes
            .append('text').attr('class', 'node-label')
            .style('opacity', n => Drawing.nodeBigEnoughForLabel(viewer, n) ? 1 : 0)
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


    function resizeVisualization(viewer: Types.Viewer, newWidth: number, newHeight: number) {
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

    function selectAtIndex<E extends d3.BaseType, D, PE extends d3.BaseType, PD>(selection: d3.Selection<E, D, PE, PD>, index: number): d3.Selection<E, D, null, undefined> {
        let s = d3.select(selection.nodes()[index]);
        return s as d3.Selection<E, D, null, undefined>;
    }

    function createNodeTooltip(node: Dag.Node): string | null {
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
        } else {
            return null;
        }
    }

    function applyFiltering(viewer: Types.Viewer, occurrenceThreshold: number): void {
        viewer.settings.occurrenceThreshold = occurrenceThreshold;
        Dag.filterDagAndAddLevels(viewer.data, occurrenceThreshold);
        refreshVisualization(viewer);
    }

    function applyColors(viewer: Types.Viewer, colorBy: Enums.ColorMethod, transition = true): void {
        viewer.settings.colorMethod = colorBy;
        viewer.data.nodes.forEach((n, i) => {
            let col: d3.RGBColor;
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
            }
            n.visual.fill = col.hex();
            n.visual.stroke = col.darker().hex();
        });
        Drawing.recolor(viewer, transition);
    }

    function applyShapes(viewer: Types.Viewer, shapeMethod: Enums.ShapeMethod): void {
        viewer.settings.shapeMethod = shapeMethod;
        setDataToViewer(viewer, viewer.data);
    }


    function readTextFile(file: string, callback: any) {
        var rawFile = new XMLHttpRequest();
        rawFile.overrideMimeType("application/json");
        rawFile.open("GET", file, true);
        rawFile.onreadystatechange = function () {
            if (rawFile.readyState === 4 && rawFile.status == 200) {
                callback(rawFile.responseText);
            } else {
                console.log('XMLHttpRequest to ' + file + ' failed.');
                console.log(rawFile);
            }
        }
        rawFile.send(null);
    }

}