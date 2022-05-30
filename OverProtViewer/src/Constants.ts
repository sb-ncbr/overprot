import * as d3 from 'd3';
import { Colors } from './Colors';
import { Enums } from './Enums';


export namespace Constants {

    export const CANVAS_HEIGHT = 300;
    export const CANVAS_WIDTH = 1200;

    export const ZOOM_STEP_RATIO = 1.25;
    export const ZOOM_STEP_RATIO_MOUSE = ZOOM_STEP_RATIO;
    export const SHIFT_STEP_RELATIVE = 0.2;
    export const MAX_EMPTY_X_MARGIN = 0.0;  // relative to screen
    export const MAX_EMPTY_Y_MARGIN = 0.0;  // relative to screen
    export const MAX_X_ZOOMOUT = 1 / (1 - 2 * MAX_EMPTY_X_MARGIN);
    export const MAX_Y_ZOOMOUT = 1 / (1 - 2 * MAX_EMPTY_Y_MARGIN);

    export const TRANSITION_DURATION = d3.transition().duration();  //ms
    export const TOOLTIP_DELAY = 600;  //ms
    export const MOUSE_HOLD_BEHAVIOR_INITIAL_SLEEP_TIME = 500;  //ms
    export const MOUSE_HOLD_BEHAVIOR_STEP_SLEEP_TIME = 50;  //ms
    export const TEMPORARY_CENTRAL_MESSAGE_TIMEOUT = 1000;  //ms

    export const TOOLTIP_OFFSET = {x: 12, y: 0};  // position of tooltip relative to cursor

    export const NODE_STROKE_WIDTH = 1;
    export const NODE_STROKE_WIDTH_HIGHLIGHTED = 3;

    export const EDGE_COLOR = '#808080';
    export const NODE_FILL = Colors.NEUTRAL_COLOR;
    export const NODE_STROKE = Colors.NEUTRAL_COLOR.darker();
    export const NODE_LABEL_COLOR = 'black';

    export const MINIMAL_WIDTH_FOR_SSE_LABEL = 20;

    export const HEATMAP_MIDDLE_VALUE = 5;
    export const DEFAULT_OCCURRENCE_THRESHOLD = 0.2;
    export const DEFAULT_BETA_CONNECTIVITY_VISIBILITY = true;
    export const DEFAULT_LAYOUT_METHOD = Enums.LayoutMethod.New;

    export const DEFAULT_COLOR_METHOD = Enums.ColorMethod.Sheet;
    export const DEFAULT_SHAPE_METHOD = Enums.ShapeMethod.Rectangle;

    export const DEFAULT_SHOW_LABELS = true;
    export const DEFAULT_SHOW_LEGEND = true;

    export const DEFAULT_DISPATCH_EVENTS = false;
    export const DEFAULT_LISTEN_EVENTS = false;


    //#region measurements in the world
    export const LENGTH_SCALE = 4;  // width of 1 residue in the world
    export const OCCURRENCE_SCALE = 100;  // height of occurrence 1.0 (100%) in the world

    export const FLOOR_HEIGHT = 1.25 * OCCURRENCE_SCALE;
    export const TOP_MARGIN = 0.35 * OCCURRENCE_SCALE;  // 0.25 * OCCURRENCE_SCALE;
    export const BOTTOM_MARGIN = 0.35 * OCCURRENCE_SCALE;  // 0.25 * OCCURRENCE_SCALE;
    export const LEFT_MARGIN = 4 * LENGTH_SCALE;
    export const RIGHT_MARGIN = 4 * LENGTH_SCALE;

    export const GAP_LENGTH = 3 * LENGTH_SCALE;
    export const KNOB_LENGTH = 1 * LENGTH_SCALE;

    export const ARC_MAX_DEVIATION = 0.5*OCCURRENCE_SCALE + Math.min(TOP_MARGIN, BOTTOM_MARGIN) - 0.1 * OCCURRENCE_SCALE;
    export const ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH = 0.001;  // slightly increasing ellipse major semiaxis provides smaller angle of arc ends
    export const ARC_MAX_MINOR = ARC_MAX_DEVIATION / (1 - Math.sqrt(1 - (1 / (1 + 2*ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH))**2));  // for elliptical arcs with extra major
    export const ARC_SMART_DEVIATION_PARAM_WRT_WORLD_WIDTH = 0.2;  // ARC_SMART_DEVIATION_PARAM = distance for which the arc deviation is 1/2*ARC_MAX_DEVIATION (for circular arcs)
    //#endregion

    export const HELIX_TYPE = 'h';
    export const STRAND_TYPE = 'e';

    export const HANGING_TEXT_OFFSET = 5;

    export const RESET_SYMBOL = '&#x27F3;';
    
    export const OPEN_POPUP_SYMBOL = ' &#x25BE;';

    export const EVENT_PREFIX = 'PDB.overprot.'
    // Outbound events (dispatched by the viewer):
    export const EVENT_TYPE_SELECT = 'select';
    export const EVENT_TYPE_HOVER = 'hover';
    // Inbound events (listened to by the viewer):
    export const EVENT_TYPE_DO_SELECT = 'do.select';
    export const EVENT_TYPE_DO_HOVER = 'do.hover';

    export const ICON_LEGEND = '<svg viewBox="0 0 100 100"><path d="M18,21 h12 v12 h-12 z M37,21 h45 v12 h-45 z M18,42 h12 v12 h-12 z M37,42 h45 v12 h-45 z M18,63 h12 v12 h-12 z M37,63 h45 v12 h-45 z"></path></svg>';
    export const ICON_LEGEND_CHECKED = '<svg viewBox="0 0 100 100"><path d="M18,21 h12 v12 h-12 z M37,21 h45 v12 h-45 z M18,42 h12 v12 h-12 z M37,42 h45 v12 h-45 z M18,63 h12 v12 h-12 z M37,63 h45 v12 h-45 z"></path>' +
    '<path erase style="stroke-width: 17px" d="M58,62 l12,22 h1 l22,-34"></path>'+
    '<path nofill style="stroke-width: 7px;" d="M60,66 l10,18 h1 l20,-30"></path>'+
    '</svg>';

    export const ICON_LABELS = '<svg viewBox="0 0 100 100"><path d="M20,80 L44,20 H56 L80,80 L70,80 L64,65 L36,65 L30,80 z M40,55 L50,30 L60,55 z"></path></svg>';
    export const ICON_LABELS_CHECKED = '<svg viewBox="0 0 100 100"><path d="M20,80 L44,20 H56 L80,80 L70,80 L64,65 L36,65 L30,80 z M40,55 L50,30 L60,55 z"></path>'+
    '<path erase style="stroke-width: 17px" d="M58,62 l12,22 h1 l22,-34"></path>'+
    '<path nofill style="stroke-width: 7px;" d="M60,66 l10,18 h1 l20,-30"></path>'+
    '</svg>';

    export const ICON_BETA_CONNECTIVITY = '<svg viewBox="0 0 100 100"><path nofill style="stroke-width: 7px;" d="M20,65 A30,40 0 0,1 80,65"></path></svg>';
    export const ICON_BETA_CONNECTIVITY_CHECKED = '<svg viewBox="0 0 100 100"><path nofill style="stroke-width: 7px;" d="M20,65 A30,40 0 0,1 80,65"></path>'+
    '<path erase style="stroke-width: 17px" d="M58,62 l12,22 h1 l22,-34"></path>'+
    '<path nofill style="stroke-width: 7px;" d="M60,66 l10,18 h1 l20,-30"></path>'+
    '</svg>';

    export const ICON_PLUS = '<svg viewBox="0 0 100 100"><path d="M25,45 H45 V25 H55 V45 H75 V55 H55 V75 H45 V55 H25 z"></path></svg>';
    export const ICON_MINUS = '<svg viewBox="0 0 100 100"><path d="M25,45 H75 V55 H25 z"></path></svg>';
    export const ICON_RESET = '<svg viewBox="0 0 100 100"><path d="M50,25 A25,25,0,1,0,75,50 H65 A15,15,0,1,1,50,35 V47 L70,31 L50,15 z"></path></svg>';
    export const ICON_CAMERA = '<svg viewBox="0 0 100 100"><path d="' + 
        'M15,30 H32 L40,22 H60 L68,30 H85 V72 H15 z ' +  // body
        'M34,50 A16,16,0,0,0,66,50 A16,16,0,0,0,34,50 z ' +  // around lens
        'M40,50 A10,10,0,0,0,60,50 A10,10,0,0,0,40,50 z ' +  // lens
        'M80,35 V41 H72 V35 z ' +  // window
        '"></path></svg>';
}