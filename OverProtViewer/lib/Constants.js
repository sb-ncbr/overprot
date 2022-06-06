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
    Constants.NODE_LABEL_COLOR = 'black';
    Constants.MINIMAL_WIDTH_FOR_SSE_LABEL = 20;
    Constants.HEATMAP_MIDDLE_VALUE = 5;
    Constants.DEFAULT_OCCURRENCE_THRESHOLD = 0.2;
    Constants.DEFAULT_BETA_CONNECTIVITY_VISIBILITY = true;
    Constants.DEFAULT_LAYOUT_METHOD = Enums.LayoutMethod.New;
    Constants.DEFAULT_COLOR_METHOD = Enums.ColorMethod.Sheet;
    Constants.DEFAULT_SHAPE_METHOD = Enums.ShapeMethod.Rectangle;
    Constants.DEFAULT_SHOW_LABELS = true;
    Constants.DEFAULT_SHOW_LEGEND = true;
    Constants.DEFAULT_DISPATCH_EVENTS = false;
    Constants.DEFAULT_LISTEN_EVENTS = false;
    Constants.DEFAULT_SAVE_NAME = 'overprot';
    //#region measurements in the world
    Constants.LENGTH_SCALE = 4; // width of 1 residue in the world
    Constants.OCCURRENCE_SCALE = 100; // height of occurrence 1.0 (100%) in the world
    Constants.FLOOR_HEIGHT = 1.25 * Constants.OCCURRENCE_SCALE;
    Constants.TOP_MARGIN = 0.35 * Constants.OCCURRENCE_SCALE; // 0.25 * OCCURRENCE_SCALE;
    Constants.BOTTOM_MARGIN = 0.35 * Constants.OCCURRENCE_SCALE; // 0.25 * OCCURRENCE_SCALE;
    Constants.LEFT_MARGIN = 4 * Constants.LENGTH_SCALE;
    Constants.RIGHT_MARGIN = 4 * Constants.LENGTH_SCALE;
    Constants.GAP_LENGTH = 3 * Constants.LENGTH_SCALE;
    Constants.KNOB_LENGTH = 1 * Constants.LENGTH_SCALE;
    Constants.ARC_MAX_DEVIATION = 0.5 * Constants.OCCURRENCE_SCALE + Math.min(Constants.TOP_MARGIN, Constants.BOTTOM_MARGIN) - 0.1 * Constants.OCCURRENCE_SCALE;
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
    Constants.SHOW_TYPE_GLYPHS = false; // debug
    Constants.ICON_LEGEND = '<svg viewBox="0 0 100 100"><path d="M18,21 h12 v12 h-12 z M37,21 h45 v12 h-45 z M18,42 h12 v12 h-12 z M37,42 h45 v12 h-45 z M18,63 h12 v12 h-12 z M37,63 h45 v12 h-45 z"></path></svg>';
    Constants.ICON_LEGEND_CHECKED = '<svg viewBox="0 0 100 100"><path d="M18,21 h12 v12 h-12 z M37,21 h45 v12 h-45 z M18,42 h12 v12 h-12 z M37,42 h45 v12 h-45 z M18,63 h12 v12 h-12 z M37,63 h45 v12 h-45 z"></path>' +
        '<path erase style="stroke-width: 17px" d="M58,62 l12,22 h1 l22,-34"></path>' +
        '<path nofill style="stroke-width: 7px;" d="M60,66 l10,18 h1 l20,-30"></path>' +
        '</svg>';
    Constants.ICON_LABELS = '<svg viewBox="0 0 100 100"><path d="M20,80 L44,20 H56 L80,80 L70,80 L64,65 L36,65 L30,80 z M40,55 L50,30 L60,55 z"></path></svg>';
    Constants.ICON_LABELS_CHECKED = '<svg viewBox="0 0 100 100"><path d="M20,80 L44,20 H56 L80,80 L70,80 L64,65 L36,65 L30,80 z M40,55 L50,30 L60,55 z"></path>' +
        '<path erase style="stroke-width: 17px" d="M58,62 l12,22 h1 l22,-34"></path>' +
        '<path nofill style="stroke-width: 7px;" d="M60,66 l10,18 h1 l20,-30"></path>' +
        '</svg>';
    Constants.ICON_BETA_CONNECTIVITY = '<svg viewBox="0 0 100 100"><path nofill style="stroke-width: 7px;" d="M20,65 A30,40 0 0,1 80,65"></path></svg>';
    Constants.ICON_BETA_CONNECTIVITY_CHECKED = '<svg viewBox="0 0 100 100"><path nofill style="stroke-width: 7px;" d="M20,65 A30,40 0 0,1 80,65"></path>' +
        '<path erase style="stroke-width: 17px" d="M58,62 l12,22 h1 l22,-34"></path>' +
        '<path nofill style="stroke-width: 7px;" d="M60,66 l10,18 h1 l20,-30"></path>' +
        '</svg>';
    Constants.ICON_PLUS = '<svg viewBox="0 0 100 100"><path d="M25,45 H45 V25 H55 V45 H75 V55 H55 V75 H45 V55 H25 z"></path></svg>';
    Constants.ICON_MINUS = '<svg viewBox="0 0 100 100"><path d="M25,45 H75 V55 H25 z"></path></svg>';
    Constants.ICON_RESET = '<svg viewBox="0 0 100 100"><path d="M50,25 A25,25,0,1,0,75,50 H65 A15,15,0,1,1,50,35 V47 L70,31 L50,15 z"></path></svg>';
    Constants.ICON_CAMERA = '<svg viewBox="0 0 100 100"><path d="' +
        'M15,30 H32 L40,22 H60 L68,30 H85 V72 H15 z ' + // body
        'M34,50 A16,16,0,0,0,66,50 A16,16,0,0,0,34,50 z ' + // around lens
        'M40,50 A10,10,0,0,0,60,50 A10,10,0,0,0,40,50 z ' + // lens
        'M80,35 V41 H72 V35 z ' + // window
        '"></path></svg>';
})(Constants || (Constants = {}));
//# sourceMappingURL=Constants.js.map