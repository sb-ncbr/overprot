import { Types } from './Types';
export declare namespace Controls {
    export type NameValueTooltip<T> = [string, T, string | null];
    type ControlBase = {
        viewer: Types.Viewer;
        id: string | null;
        children: Control[];
        div: Types.D3Selection;
        tooltip: string | null;
        show: (parentDiv: Types.D3Selection) => void;
    };
    type Control = {
        base: ControlBase;
    };
    type ControlPanel = {
        base: ControlBase;
    };
    type Button = {
        base: ControlBase;
        text: string;
        square: boolean;
        icon: boolean;
        onClick: () => any;
    };
    type Popup = {
        base: ControlBase;
        headButton: Button;
        autocollapse: boolean;
    };
    type Listbox<T> = {
        base: ControlBase;
        namesValuesTooltips: [string, T, string | null][];
        selectedValue: T;
        onSelect: (value: T) => any;
    };
    type Slider = {
        base: ControlBase;
        minValue: number;
        maxValue: number;
        step: number;
        selectedValue: number;
        minValueLabel: string;
        maxValueLabel: string;
        onMove: (value: number) => any;
        onRelease: (value: number) => any;
    };
    export function newControlPanel(viewer: Types.Viewer, id: string | null, tooltip: string | null): ControlPanel;
    export function addToControlPanel(panel: ControlPanel, child: Control): void;
    export function newButton(viewer: Types.Viewer, id: string | null, text: string, square: boolean, icon: boolean, onClick: () => any, tooltip: string | null): Button;
    export function changeButtonText(button: Button, newText: string): void;
    export function newToggleButton(viewer: Types.Viewer, id: string | null, text: string | [string, string], square: boolean, icon: boolean, selectedValue: boolean, onSelect: (value: boolean) => any, tooltip: string | null): Button;
    export function newPopup(viewer: Types.Viewer, id: string | null, text: string, autocollapse: boolean, tooltip: string | null): Popup;
    export function addToPopup(popup: Popup, child: Control): void;
    export function newListbox<T>(viewer: Types.Viewer, id: string | null, namesValuesTooltips: NameValueTooltip<T>[], selectedValue: T, onSelect: (value: T) => any, tooltip: string | null): Listbox<T>;
    export function newDropdownList<T>(viewer: Types.Viewer, id: string | null, text: string, namesValuesTooltips: NameValueTooltip<T>[], selectedValue: T, onSelect: (value: T) => any, tooltip: string | null, autocollapse?: boolean, collapseOnSelect?: boolean): Popup;
    export function newSlider(viewer: Types.Viewer, id: string | null, minValue: number, maxValue: number, step: number, selectedValue: number, minValueLabel: string | null, maxValueLabel: string | null, // if null, minValue and maxValue will be shown; if '', no label will be shown
    onMove: (value: number) => any, onRelease: (value: number) => any, tooltip: string | null): Slider;
    export function showSlider(slider: Slider, parentDiv: Types.D3Selection): void;
    export function newPopupSlider(viewer: Types.Viewer, id: string | null, textPrefix: string, textSuffix: string, minValue: number, maxValue: number, step: number, selectedValue: number, minValueLabel: string | null, maxValueLabel: string | null, // if null, minValue and maxValue will be shown; if '', no label will be shown
    onMove: (value: number) => any, onRelease: (value: number) => any, tooltip: string | null, autocollapse?: boolean): Popup;
    export {};
}
