import * as d3 from 'd3';
import { Dag } from './Dag';
import { Geometry } from './Geometry';
import { Enums } from './Enums';
import { Constants } from './Constants';

export namespace Types {

    export type D3Selection = d3.Selection<d3.BaseType, any, d3.BaseType, any>;
    export type D3Transition = d3.Transition<d3.BaseType, any, d3.BaseType, any>;

    export type Viewer = {
        id: string,
        internalId: string,
        d3viewer: d3.Selection<HTMLElement, unknown, null, undefined>,
        mainDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>,
        guiDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>,
        canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>,
        data: Dag.Dag,
        world: Geometry.Rectangle,
        visWorld: Geometry.Rectangle,
        screen: Geometry.Rectangle,
        zoom: Geometry.ZoomInfo,
        settings: Settings,
        nodeMap: Map<string, SVGElement>,
        ladderMap: TupleMap<string, SVGElement>,
    };

    export function newViewer(
        id: string,
        internalId: string,
        d3viewer: d3.Selection<HTMLElement, unknown, null, undefined>,
        d3mainDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>,
        d3guiDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>,
        d3canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>,
        settings: Settings | null = null): Viewer {
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
            settings: settings ?? newSettings(),
            nodeMap: new Map(),
            ladderMap: new TupleMap(),
        };
    };

    export type Settings = {
        file: string,
        width: number,
        height: number,
        colorMethod: Enums.ColorMethod,
        shapeMethod: Enums.ShapeMethod,
        layoutMethod: Enums.LayoutMethod,
        betaConnectivityVisibility: boolean,
        occurrenceThreshold: number,
        dispatchEvents: boolean,
        listenEvents: boolean
    };

    export function newSettings(): Settings {
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

    export function newSettingsFromHTMLElement(element: HTMLElement): Settings {
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
            let attributeName = element.attributes.item(i)!.name;
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
        }
        const shapeMethodDictionary = {
            'rectangle': Enums.ShapeMethod.Rectangle,
            'symcdf': Enums.ShapeMethod.SymCdf,
        }
        const layoutMethodDictionary = {
            'old': Enums.LayoutMethod.Old,
            'new': Enums.LayoutMethod.New,
        }
        const booleanDictionary = {
            'on': true,
            'off': false,
            'true': true,
            'false': false,
            '1': true,
            '0': false,
        }
        return {
            file: d3element.attr('file') ?? '',
            height: parseIntAttribute('height', d3element.attr('height'), Constants.CANVAS_HEIGHT),
            width: parseIntAttribute('width', d3element.attr('width'), Constants.CANVAS_WIDTH),
            colorMethod: parseEnumAttribute('color-method', d3element.attr('color-method'), colorMethodDictionary, Constants.DEFAULT_COLOR_METHOD),
            shapeMethod: parseEnumAttribute('shape-method', d3element.attr('shape-method'), shapeMethodDictionary, Constants.DEFAULT_SHAPE_METHOD),
            layoutMethod: parseEnumAttribute('layout-method', d3element.attr('layout-method'), layoutMethodDictionary, Constants.DEFAULT_LAYOUT_METHOD),
            betaConnectivityVisibility: parseEnumAttribute('beta-connectivity', d3element.attr('beta-connectivity'), booleanDictionary, Constants.DEFAULT_BETA_CONNECTIVITY_VISIBILITY),
            occurrenceThreshold: parseFloatAttribute('occurrence-threshold', d3element.attr('occurrence-threshold'), Constants.DEFAULT_OCCURRENCE_THRESHOLD, [0, 1], true),
            dispatchEvents: parseEnumAttribute('dispatch-events', d3element.attr('dispatch-events'), booleanDictionary, Constants.DEFAULT_DISPATCH_EVENTS),
            listenEvents: parseEnumAttribute('listen-events', d3element.attr('listen-events'), booleanDictionary, Constants.DEFAULT_LISTEN_EVENTS),
        }
    }

    function parseEnumAttribute<T>(attributeName: string, attributeValue: string | null, dict: { [methodName: string]: T }, defaultValue: T): T {
        if (attributeValue === null) {
            return defaultValue;
        }
        else if (dict[attributeValue] !== undefined) {
            return dict[attributeValue];
        } else {
            console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Allowed values: ${Object.keys(dict)}.`);
            return defaultValue;
        }
    }

    function parseIntAttribute(attributeName: string, attributeValue: string | null, defaultValue: number, minMaxLimits: number[] = []): number {
        if (attributeValue === null) {
            return defaultValue;
        }
        else {
            let value = parseInt(attributeValue);
            if (isNaN(value)) {
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be an integer.`);
                return defaultValue;
            } else if (minMaxLimits.length >= 2 && (value < minMaxLimits[0] || value > minMaxLimits[1])) {
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be an integer between ${minMaxLimits[0]} and ${minMaxLimits[1]}.`);
                return defaultValue;
            } else {
                return value;
            }
        }
    }

    function parseFloatAttribute(attributeName: string, attributeValue: string | null, defaultValue: number, minMaxLimits: number[] = [], allowPercentage = false): number {
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
            } else if (minMaxLimits.length >= 2 && (value < minMaxLimits[0] || value > minMaxLimits[1])) {
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be a float between ${minMaxLimits[0]} and ${minMaxLimits[1]}.`);
                return defaultValue;
            } else {
                return value;
            }
        }
    }

    export class TupleMap<K, V> {
        /* Map where keys can be tuples. */
        map: Map<K, TupleMap<K, V>> | undefined;
        value: V | undefined;

        constructor() {
            this.map = undefined;
            this.value = undefined;
        }
        get(key: K[]): V | undefined {
            let currentTM: TupleMap<K, V> | undefined = this;
            for (const k of key) {
                currentTM = currentTM.map?.get(k);
                if (currentTM == undefined) return undefined;
            }
            return currentTM.value;
        }
        set(key: K[], value: V): void {
            let currentTM: TupleMap<K, V> = this;
            for (const k of key) {
                if (!currentTM.map) {
                    currentTM.map = new Map();
                }
                if (!currentTM.map.has(k)) {
                    currentTM.map.set(k, new TupleMap());
                }
                currentTM = currentTM.map.get(k)!;
            }
            currentTM.value = value;
        }
        entries(): [K[], V][] {
            let outList: [K[], V][] = [];
            this.collectEntries([], outList);
            return outList;
        }
        private collectEntries(prefix: K[], outList: [K[], V][]): void {
            if (this.value != undefined) {
                outList.push([[...prefix], this.value]);
            }
            if (this.map != undefined) {
                for(const [k, sub] of this.map.entries()){
                    prefix.push(k);
                    sub.collectEntries(prefix, outList);
                    prefix.pop();
                }
            }
        }
    }



}