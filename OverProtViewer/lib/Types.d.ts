import * as d3 from 'd3';
import { Dag } from './Dag';
import { Geometry } from './Geometry';
import { Enums } from './Enums';
export declare namespace Types {
    type D3Selection = d3.Selection<d3.BaseType, any, d3.BaseType, any>;
    type D3Transition = d3.Transition<d3.BaseType, any, d3.BaseType, any>;
    type Viewer = {
        id: string;
        uniqueId: string;
        mainDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>;
        guiDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>;
        canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>;
        data: Dag.Dag;
        world: Geometry.Rectangle;
        visWorld: Geometry.Rectangle;
        screen: Geometry.Rectangle;
        zoom: Geometry.ZoomInfo;
        settings: Settings;
    };
    function newViewer(id: string, uniqueId: string, d3mainDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>, d3guiDiv: d3.Selection<HTMLDivElement, unknown, null, undefined>, d3canvas: d3.Selection<SVGSVGElement, any, d3.BaseType, any>, settings?: Settings | null): Viewer;
    type Settings = {
        file: string;
        width: number;
        height: number;
        colorMethod: Enums.ColorMethod;
        shapeMethod: Enums.ShapeMethod;
        layoutMethod: Enums.LayoutMethod;
        betaConnectivityVisibility: boolean;
        occurrenceThreshold: number;
    };
    function newSettings(): Settings;
    function newSettingsFromHTMLElement(element: HTMLElement): Settings;
}
