import * as d3 from 'd3';
import { RGBColor } from 'd3';
export declare namespace Colors {
    const NEUTRAL_COLOR: d3.RGBColor, COLOR_SCHEME: d3.RGBColor[];
    const NEUTRAL_DARK: d3.RGBColor;
    function bySseType(type: string): RGBColor;
    function byIndex0(i: number): RGBColor;
    function byIndex1(i: number): RGBColor;
    /** Maps 0 to the coolest, max to the hottest. */
    function byLinHeatmap(value: number, max: number): RGBColor;
    /** Maps 0 to the coolest, middle to the middle of the scale, infinity to the hottest. */
    function byExpHeatmap(value: number, middle: number): RGBColor;
}
