import * as d3 from 'd3';
import { Colors } from './Colors';
import { Enums } from './Enums';
export var Constants;
(function (Constants) {
    Constants.CANVAS_HEIGHT = 300;
    Constants.CANVAS_WIDTH = 1200;
    Constants.ZOOM_STEP_RATIO = 1.25;
    Constants.ZOOM_STEP_RATIO_MOUSE = Constants.ZOOM_STEP_RATIO;
    Constants.SHIFT_STEP_RELATIVE = 0.2;
    Constants.MAX_EMPTY_X_MARGIN = 0.0; // relative to screen
    Constants.MAX_EMPTY_Y_MARGIN = 0.0; // relative to screen
    Constants.MAX_X_ZOOMOUT = 1 / (1 - 2 * Constants.MAX_EMPTY_X_MARGIN);
    Constants.MAX_Y_ZOOMOUT = 1 / (1 - 2 * Constants.MAX_EMPTY_Y_MARGIN);
    Constants.TRANSITION_DURATION = d3.transition().duration(); //ms
    Constants.TOOLTIP_DELAY = 600; //ms
    Constants.MOUSE_HOLD_BEHAVIOR_INITIAL_SLEEP_TIME = 500; //ms
    Constants.MOUSE_HOLD_BEHAVIOR_STEP_SLEEP_TIME = 50; //ms
    Constants.TEMPORARY_CENTRAL_MESSAGE_TIMEOUT = 1000; //ms
    Constants.TOOLTIP_OFFSET = { x: 12, y: 0 }; // position of tooltip relative to cursor
    Constants.NODE_STROKE_WIDTH = 1;
    Constants.NODE_STROKE_WIDTH_HIGHLIGHTED = 3;
    Constants.EDGE_COLOR = '#808080';
    Constants.NODE_FILL = Colors.NEUTRAL_COLOR;
    Constants.NODE_STROKE = Colors.NEUTRAL_COLOR.darker();
    Constants.MINIMAL_WIDTH_FOR_SSE_LABEL = 20;
    Constants.HEATMAP_MIDDLE_VALUE = 5;
    Constants.DEFAULT_OCCURRENCE_THRESHOLD = 0.2;
    Constants.DEFAULT_BETA_CONNECTIVITY_VISIBILITY = true;
    Constants.DEFAULT_LAYOUT_METHOD = Enums.LayoutMethod.New;
    Constants.DEFAULT_COLOR_METHOD = Enums.ColorMethod.Sheet;
    Constants.DEFAULT_SHAPE_METHOD = Enums.ShapeMethod.Rectangle;
    Constants.DEFAULT_DISPATCH_EVENTS = false;
    Constants.DEFAULT_LISTEN_EVENTS = false;
    //#region measurements in the world
    Constants.LENGTH_SCALE = 4; // width of 1 residue in the world
    Constants.OCCURRENCE_SCALE = 100; // height of occurrence 1.0 (100%) in the world
    Constants.FLOOR_HEIGHT = 1.25 * Constants.OCCURRENCE_SCALE; // 1.5 * OCCURRENCE_SCALE;
    Constants.TOP_MARGIN = 0.25 * Constants.OCCURRENCE_SCALE;
    Constants.BOTTOM_MARGIN = 0.25 * Constants.OCCURRENCE_SCALE;
    Constants.LEFT_MARGIN = 4 * Constants.LENGTH_SCALE;
    Constants.RIGHT_MARGIN = 4 * Constants.LENGTH_SCALE;
    Constants.GAP_LENGTH = 3 * Constants.LENGTH_SCALE;
    Constants.KNOB_LENGTH = 1 * Constants.LENGTH_SCALE;
    Constants.ARC_MAX_DEVIATION = 0.5 * Constants.OCCURRENCE_SCALE + Math.min(Constants.TOP_MARGIN, Constants.BOTTOM_MARGIN);
    Constants.ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH = 0.001; // slightly increasing ellipse major semiaxis provides smaller angle of arc ends
    Constants.ARC_MAX_MINOR = Constants.ARC_MAX_DEVIATION / (1 - Math.sqrt(1 - Math.pow((1 / (1 + 2 * Constants.ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH)), 2))); // for elliptical arcs with extra major
    Constants.ARC_SMART_DEVIATION_PARAM_WRT_WORLD_WIDTH = 0.2; // ARC_SMART_DEVIATION_PARAM = distance for which the arc deviation is 1/2*ARC_MAX_DEVIATION (for circular arcs)
    //#endregion
    Constants.HELIX_TYPE = 'h';
    Constants.STRAND_TYPE = 'e';
    Constants.HANGING_TEXT_OFFSET = 5;
    Constants.RESET_SYMBOL = '&#x27F3;';
    Constants.OPEN_POPUP_SYMBOL = ' &#x25BE;';
    Constants.EVENT_PREFIX = 'PDB.overprot.';
    // Outbound events (dispatched by the viewer):
    Constants.EVENT_TYPE_SELECT = 'select';
    Constants.EVENT_TYPE_HOVER = 'hover';
    // Inbound events (listened to by the viewer):
    Constants.EVENT_TYPE_DO_SELECT = 'do.select';
    Constants.EVENT_TYPE_DO_HOVER = 'do.hover';
})(Constants || (Constants = {}));
//# sourceMappingURL=Constants.js.map