import * as d3 from 'd3';
import { RGBColor } from 'd3';

export namespace Colors {
    export const { neutral: NEUTRAL_COLOR, scheme: COLOR_SCHEME } = makeScheme();

    function makeScheme(): { neutral: RGBColor, scheme: RGBColor[] } {
        let scheme = d3.schemeCategory10.map(str => d3.rgb(str));
        let neutral = scheme.splice(7, 1);
        return { neutral: neutral[0], scheme: scheme };
    }

    export function bySseType(type: string): RGBColor {
        type = type.toLowerCase();
        if (type == 'h') return NEUTRAL_COLOR;
        else if (type == 'e') return COLOR_SCHEME[0];
        else return COLOR_SCHEME[1];
    }

    export function byIndex0(i: number): RGBColor {
        return COLOR_SCHEME[i % COLOR_SCHEME.length];
    }

    export function byIndex1(i: number): RGBColor {
        if (i >= 1) return COLOR_SCHEME[(i - 1) % COLOR_SCHEME.length];
        else return NEUTRAL_COLOR;
    }

    /** Maps 0 to the coolest, max to the hottest. */
    export function byLinHeatmap(value: number, max: number): RGBColor {
        const x = value / max;
        return d3.rgb(d3.interpolateInferno(x));
    }

    /** Maps 0 to the coolest, middle to the middle of the scale, infinity to the hottest. */
    export function byExpHeatmap(value: number, middle: number): RGBColor {
        const x = 1 - 2 ** (-value / middle);
        return d3.rgb(d3.interpolateInferno(x));
    }
}