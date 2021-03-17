import * as d3 from 'd3';
export var Colors;
(function (Colors) {
    var _a;
    _a = makeScheme(), Colors.NEUTRAL_COLOR = _a.neutral, Colors.COLOR_SCHEME = _a.scheme;
    function makeScheme() {
        let scheme = d3.schemeCategory10.map(str => d3.rgb(str));
        let neutral = scheme.splice(7, 1);
        return { neutral: neutral[0], scheme: scheme };
    }
    function bySseType(type) {
        type = type.toLowerCase();
        if (type == 'h')
            return Colors.NEUTRAL_COLOR;
        else if (type == 'e')
            return Colors.COLOR_SCHEME[0];
        else
            return Colors.COLOR_SCHEME[1];
    }
    Colors.bySseType = bySseType;
    function byIndex0(i) {
        return Colors.COLOR_SCHEME[i % Colors.COLOR_SCHEME.length];
    }
    Colors.byIndex0 = byIndex0;
    function byIndex1(i) {
        if (i >= 1)
            return Colors.COLOR_SCHEME[(i - 1) % Colors.COLOR_SCHEME.length];
        else
            return Colors.NEUTRAL_COLOR;
    }
    Colors.byIndex1 = byIndex1;
    /** Maps 0 to the coolest, max to the hottest. */
    function byLinHeatmap(value, max) {
        const x = value / max;
        return d3.rgb(d3.interpolateInferno(x));
    }
    Colors.byLinHeatmap = byLinHeatmap;
    /** Maps 0 to the coolest, middle to the middle of the scale, infinity to the hottest. */
    function byExpHeatmap(value, middle) {
        const x = 1 - Math.pow(2, (-value / middle));
        return d3.rgb(d3.interpolateInferno(x));
    }
    Colors.byExpHeatmap = byExpHeatmap;
})(Colors || (Colors = {}));
//# sourceMappingURL=Colors.js.map