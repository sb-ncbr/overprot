import * as d3 from 'd3';
import { Dag } from './Dag';
import { Geometry } from './Geometry';
import { Enums } from './Enums';
import { Constants } from './Constants';
export var Types;
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
    ;
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
            showLabels: Constants.DEFAULT_SHOW_LABELS,
            showLegend: Constants.DEFAULT_SHOW_LEGEND,
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
            'show-labels', 'show-legend',
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
            showLabels: parseEnumAttribute('show-labels', d3element.attr('show-labels'), booleanDictionary, Constants.DEFAULT_SHOW_LABELS),
            showLegend: parseEnumAttribute('show-legend', d3element.attr('show-legend'), booleanDictionary, Constants.DEFAULT_SHOW_LEGEND),
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
//# sourceMappingURL=Types.js.map