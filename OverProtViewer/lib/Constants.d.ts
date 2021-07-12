import * as d3 from 'd3';
import { Enums } from './Enums';
export declare namespace Constants {
    const CANVAS_HEIGHT = 300;
    const CANVAS_WIDTH = 1500;
    const ZOOM_STEP_RATIO = 1.25;
    const ZOOM_STEP_RATIO_MOUSE = 1.25;
    const SHIFT_STEP_RELATIVE = 0.2;
    const MAX_EMPTY_X_MARGIN = 0;
    const MAX_EMPTY_Y_MARGIN = 0;
    const MAX_X_ZOOMOUT: number;
    const MAX_Y_ZOOMOUT: number;
    const TRANSITION_DURATION: number;
    const TOOLTIP_DELAY = 600;
    const MOUSE_HOLD_BEHAVIOR_INITIAL_SLEEP_TIME = 500;
    const MOUSE_HOLD_BEHAVIOR_STEP_SLEEP_TIME = 50;
    const TEMPORARY_CENTRAL_MESSAGE_TIMEOUT = 1000;
    const TOOLTIP_OFFSET: {
        x: number;
        y: number;
    };
    const NODE_STROKE_WIDTH = 1;
    const NODE_STROKE_WIDTH_HIGHLIGHTED = 3;
    const EDGE_COLOR = "#808080";
    const NODE_FILL: d3.RGBColor;
    const NODE_STROKE: d3.RGBColor;
    const MINIMAL_WIDTH_FOR_SSE_LABEL = 20;
    const HEATMAP_MIDDLE_VALUE = 5;
    const DEFAULT_OCCURRENCE_THRESHOLD = 0.2;
    const DEFAULT_BETA_CONNECTIVITY_VISIBILITY = true;
    const DEFAULT_LAYOUT_METHOD = Enums.LayoutMethod.New;
    const DEFAULT_COLOR_METHOD = Enums.ColorMethod.Sheet;
    const DEFAULT_SHAPE_METHOD = Enums.ShapeMethod.Rectangle;
    const DEFAULT_DISPATCH_EVENTS = false;
    const DEFAULT_LISTEN_EVENTS = false;
    const LENGTH_SCALE = 4;
    const OCCURRENCE_SCALE = 100;
    const FLOOR_HEIGHT: number;
    const TOP_MARGIN: number;
    const BOTTOM_MARGIN: number;
    const LEFT_MARGIN: number;
    const RIGHT_MARGIN: number;
    const GAP_LENGTH: number;
    const KNOB_LENGTH: number;
    const ARC_MAX_DEVIATION: number;
    const ARC_EXTRA_MAJOR_WRT_WORLD_WIDTH = 0.001;
    const ARC_MAX_MINOR: number;
    const ARC_SMART_DEVIATION_PARAM_WRT_WORLD_WIDTH = 0.2;
    const HELIX_TYPE = "h";
    const STRAND_TYPE = "e";
    const HANGING_TEXT_OFFSET = 5;
    const RESET_SYMBOL = "&#x27F3;";
    const OPEN_POPUP_SYMBOL = " &#x25BE;";
    const EVENT_PREFIX = "PDB.overprot.";
    const EVENT_TYPE_SELECT = "select";
    const EVENT_TYPE_HOVER = "hover";
    const EVENT_TYPE_DO_SELECT = "do.select";
    const EVENT_TYPE_DO_HOVER = "do.hover";
}
