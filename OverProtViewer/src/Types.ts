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
        uniqueId: string,
        mainDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>,
        guiDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>,
        canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>,
        data: Dag.Dag,
        world: Geometry.Rectangle,
        visWorld: Geometry.Rectangle,
        screen: Geometry.Rectangle,
        zoom: Geometry.ZoomInfo,
        settings: Settings
    };

    export function newViewer(
            id: string,
            uniqueId: string,
            d3mainDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>, 
            d3guiDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>, 
            d3canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>, 
            settings: Settings|null = null): Viewer {
        return {
            id: id,
            uniqueId: uniqueId,
            mainDiv: d3mainDiv,
            guiDiv: d3guiDiv,
            canvas: d3canvas,
            data: Dag.newDag(),
            world: Geometry.newRectangle(),
            visWorld: Geometry.newRectangle(),
            screen: Geometry.rectangleFromCanvas(d3canvas),
            zoom: Geometry.newZoomInfo(1, 1, 1, 1, 1),
            settings: settings ?? newSettings()
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
        occurrenceThreshold: number
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
            occurrenceThreshold: Constants.DEFAULT_OCCURRENCE_THRESHOLD
        };
    }

    export function newSettingsFromHTMLElement(element: HTMLElement): Settings {
        let MANDATORY_ATTRIBUTES = ['file'];
        let ALLOWED_ATTRIBUTES = ['id', 'file', 'width', 'height', 'color-method', 'shape-method', 'layout-method', 'beta-connectivity', 'occurrence-threshold'];
        MANDATORY_ATTRIBUTES.forEach(attributeName => {
            if (!element.hasAttribute(attributeName)){
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

        let colorMethodDictionary = {
            'uniform': Enums.ColorMethod.Uniform,
            'type': Enums.ColorMethod.Type,
            'sheet': Enums.ColorMethod.Sheet,
            'variability': Enums.ColorMethod.Stdev,
        }
        let shapeMethodDictionary = {
            'rectangle': Enums.ShapeMethod.Rectangle,
            'symcdf': Enums.ShapeMethod.SymCdf,
        }
        let layoutMethodDictionary = {
            'old': Enums.LayoutMethod.Old,
            'new': Enums.LayoutMethod.New,
        }
        let betaConnectivityDictionary = {
            'on': true,
            'off': false,
        }
        return {
            file: d3element.attr('file') ?? '',
            height: parseIntAttribute('height', d3element.attr('height'), Constants.CANVAS_HEIGHT),
            width: parseIntAttribute('width', d3element.attr('width'), Constants.CANVAS_WIDTH),
            colorMethod: parseEnumAttribute('color-method', d3element.attr('color-method'), colorMethodDictionary, Constants.DEFAULT_COLOR_METHOD),
            shapeMethod: parseEnumAttribute('shape-method', d3element.attr('shape-method'), shapeMethodDictionary, Constants.DEFAULT_SHAPE_METHOD),
            layoutMethod: parseEnumAttribute('layout-method', d3element.attr('layout-method'), layoutMethodDictionary, Constants.DEFAULT_LAYOUT_METHOD),
            betaConnectivityVisibility: parseEnumAttribute('beta-connectivity', d3element.attr('beta-connectivity'), betaConnectivityDictionary, Constants.DEFAULT_BETA_CONNECTIVITY_VISIBILITY),
            occurrenceThreshold: parseFloatAttribute('occurrence-threshold', d3element.attr('occurrence-threshold'), Constants.DEFAULT_OCCURRENCE_THRESHOLD, [0, 1], true)
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
            if (isNaN(value)){
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be an integer.`);
                return defaultValue;
            } else if (minMaxLimits.length >= 2 && (value < minMaxLimits[0] || value > minMaxLimits[1])){
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be an integer between ${minMaxLimits[0]} and ${minMaxLimits[1]}.`);
                return defaultValue;
            } else {
                return value;
            }
        }
    }

    function parseFloatAttribute(attributeName: string, attributeValue: string | null, defaultValue: number, minMaxLimits: number[] = [], allowPercentage=false): number {
        if (attributeValue === null) {
            return defaultValue;
        }
        else {
            let value = parseFloat(attributeValue);
            if (allowPercentage && attributeValue.includes('%')) {
                value *= 0.01;
            }
            if (isNaN(value)){
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be a float.`);
                return defaultValue;
            } else if (minMaxLimits.length >= 2 && (value < minMaxLimits[0] || value > minMaxLimits[1])){
                console.warn(`Attribute "${attributeName}" has invalid value "${attributeValue}". Value must be a float between ${minMaxLimits[0]} and ${minMaxLimits[1]}.`);
                return defaultValue;
            } else {
                return value;
            }
        }
    }

}