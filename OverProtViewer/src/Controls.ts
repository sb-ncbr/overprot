import * as d3 from 'd3';
import * as d3SelectionMulti from 'd3-selection-multi';

import { Constants } from './Constants';
import { Colors } from './Colors';
import { Types } from './Types';
import { Dag } from './Dag';
import { Drawing } from './Drawing';
import { Geometry } from './Geometry';
import { Enums } from './Enums';
import { BaseType, min } from 'd3';


export namespace Controls {

    export type NameValueTooltip<T> = [string, T, string | null];

    type ControlBase = { viewer: Types.Viewer, id: string | null, children: Control[], div: Types.D3Selection, tooltip: string | null, show: (parentDiv: Types.D3Selection) => void };
    
    type Control = { base: ControlBase };
    type ControlPanel = { base: ControlBase };
    type Button = { base: ControlBase, text: string, square: boolean, icon: boolean, onClick: () => any };
    type Popup = { base: ControlBase, headButton: Button, autocollapse: boolean };
    type Listbox<T> = { base: ControlBase, 
        namesValuesTooltips: [string, T, string | null][], selectedValue: T, onSelect: (value: T) => any };
    type Slider = { base: ControlBase, 
        minValue: number, maxValue: number, step: number, selectedValue: number, 
        minValueLabel: string, maxValueLabel: string,
        onMove: (value: number) => any, onRelease: (value: number) => any };


    function emptySelection(): Types.D3Selection {
        return d3.selectAll() as any;
    }

    export function newControlPanel(viewer: Types.Viewer, id: string|null, tooltip: string|null): ControlPanel {
        let panel = { 
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par: Types.D3Selection) => showControlPanel(panel, par)}
        };
        return panel;
    }
    export function addToControlPanel(panel: ControlPanel, child: Control): void {
        panel.base.children.push(child);
    }
    function showControlPanel(panel: ControlPanel, parentDiv: Types.D3Selection): void {
        panel.base.div = parentDiv.append('div').attr('class', 'control-panel').attr('id', panel.base.id!) as any;
        panel.base.children.forEach(child => child.base.show(panel.base.div));
    }

    export function newButton(viewer: Types.Viewer, id: string|null, text: string, square: boolean, icon: boolean, onClick: () => any, tooltip: string|null): Button {
        let button = { 
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par: Types.D3Selection) => showButton(button, par)}, 
            text: text, 
            square: square,
            icon: icon,
            onClick: onClick 
        };
        return button;
    }
    function showButton(button: Button, parentDiv: Types.D3Selection): void {
        button.base.div = parentDiv.append('div').attr('class', button.square ? 'button square-button' : 'button').attr('id', button.base.id!) as any;
        const clas = button.icon ? 'button-icon':'button-text';
        button.base.div.append('div').attr('class', clas).html(button.text);
        Drawing.setTooltips(button.base.viewer, button.base.div, [button.base.tooltip], false, true);
        button.base.div.on('click', button.onClick);
        button.base.div.on('dblclick', () => { d3.event.stopPropagation() });
    }
    export function changeButtonText(button: Button, newText: string): void {
        button.text = newText;
        button.base.div.select('div.button-icon,div.button-text').html(newText);
    }

    export function newPopup(viewer: Types.Viewer, id: string|null, text: string, autocollapse: boolean, tooltip: string|null): Popup {
        // console.log('newPopup');
        let popup: Popup = { 
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par: Types.D3Selection) => showPopup(popup, par)}, 
            headButton: newButton(viewer, null, text + Constants.OPEN_POPUP_SYMBOL, false, false, () => togglePopup(popup), tooltip),
            autocollapse: autocollapse 
        };
        return popup;
    }
    export function addToPopup(popup: Popup, child: Control): void {
        // console.log('addToPopup', popup, child);
        popup.base.children.push(child);
    }
    function showPopup(popup: Popup, parentDiv: Types.D3Selection): void {
        popup.base.div = parentDiv.append('div').attr('class', popup.autocollapse ? 'popup autocollapse' : 'popup').attr('id', popup.base.id!) as any;
        let headDiv = popup.base.div.append('div').attr('class', 'popup-head') as any;
        popup.base.div.data([popup]);
        showButton(popup.headButton, headDiv);
    }
    function togglePopup(popup: Popup): void {
        // console.log('togglePopup', popup);
        let expanded = popup.base.div.selectAll('div.popup-tail').size() > 0;
        if (expanded) {
            collapsePopup(popup);
        } else {
            collapseAllAutocollapsePopups(popup.base.viewer);
            expandPopup(popup);
            d3.event.stopPropagation();
        }
    }
    function expandPopup<T>(popup: Popup): void {
        // console.log('expandPopup');
        let headDiv = popup.base.div.select('div.popup-head') as any;
        let tailDiv = popup.base.div.append('div').attr('class', 'popup-tail') as any;
        popup.base.children.forEach(child => child.base.show(tailDiv));
        // let headWidth = headDiv.node().getBoundingClientRect().width;
        // let tailWidth = tailDiv.node()?.getBoundingClientRect().width;
        // if (tailWidth !== undefined && headWidth !== undefined && tailWidth < headWidth) {
        //     tailDiv.style('width', headWidth);
        // }
        if (popup.autocollapse) {
            popup.base.viewer.mainDiv.on('click.autocollapse-popups', () => collapseAllAutocollapsePopups(popup.base.viewer));
            tailDiv.on('click.autocollapse-popups', () => d3.event.stopPropagation());
        }
    }
    function collapsePopup<T>(popup: Popup): void {
        // console.log('collapsePopup');
        popup.base.div.selectAll('div.popup-tail').remove();
    }
    function collapseAllAutocollapsePopups<T>(viewer: Types.Viewer): void {
        viewer.mainDiv.selectAll('div.popup.autocollapse').each(d => collapsePopup(d as Popup));
    }
    function changePopupText(popup: Popup, newText: string): void {
        changeButtonText(popup.headButton, newText + Constants.OPEN_POPUP_SYMBOL);
    }

    export function newListbox<T>(viewer: Types.Viewer, id: string|null, namesValuesTooltips: NameValueTooltip<T>[], 
            selectedValue: T, onSelect: (value: T) => any, tooltip: string|null): Listbox<T> {
        let listbox: Listbox<T> = { 
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par: Types.D3Selection) => showListbox(listbox, par)}, 
            namesValuesTooltips: namesValuesTooltips,
            selectedValue: selectedValue,
            onSelect: onSelect
        };
        return listbox;
    }
    function showListbox<T>(listbox: Listbox<T>, parentDiv: Types.D3Selection): void {
        listbox.base.div = parentDiv.append('div').attr('class', 'listbox').attr('id', listbox.base.id!) as any;
        let itemOnClick = (nvt: NameValueTooltip<T>) => {
            listbox.selectedValue = nvt[1];
            listbox.base.div.selectAll('div.listbox-item')
                .attr('class', nvt2 => (nvt2 as NameValueTooltip<T>)[1] == listbox.selectedValue ? 'button listbox-item selected' : 'button listbox-item');
            listbox.onSelect(nvt[1]);
        };        
        listbox.base.div.selectAll().data(listbox.namesValuesTooltips)
            .enter().append('div').attr('class', kv => kv[1] == listbox.selectedValue ? 'button listbox-item selected' : 'button listbox-item')
            .on('click', itemOnClick)
            .append('div').attr('class', 'button-text').html(kv => kv[0]);
        Drawing.setTooltips(listbox.base.viewer, listbox.base.div.selectAll('div.listbox-item'), listbox.namesValuesTooltips.map(nvt => nvt[2]), false, true);
        Drawing.setTooltips(listbox.base.viewer, listbox.base.div, [listbox.base.tooltip], false, true);
    }

    export function newDropdownList<T>(viewer: Types.Viewer, id: string|null, text: string, namesValuesTooltips: NameValueTooltip<T>[], 
            selectedValue: T, onSelect: (value: T) => any, tooltip: string|null, autocollapse: boolean = true, collapseOnSelect: boolean = true): Popup {
        let popup = newPopup(viewer, id, text, autocollapse, tooltip);
        let wrappedOnSelect = collapseOnSelect ?
            (value: T) => { collapsePopup(popup); onSelect(value); }
            : onSelect;
        let listbox = newListbox(viewer, id, namesValuesTooltips, selectedValue, wrappedOnSelect, null);
        addToPopup(popup, listbox);
        return popup;
    }

    export function newSlider(viewer: Types.Viewer, id: string|null, minValue: number, maxValue: number, step: number, selectedValue: number, 
        minValueLabel: string|null, maxValueLabel: string|null, // if null, minValue and maxValue will be shown; if '', no label will be shown
        onMove: (value: number) => any, onRelease: (value: number) => any, tooltip: string|null): Slider {
        // console.log('newSlider');
        let slider: Slider = { 
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par: Types.D3Selection) => showSlider(slider, par)}, 
            minValue: minValue,
            maxValue: maxValue,
            step: step,
            selectedValue: selectedValue,
            minValueLabel: minValueLabel ?? minValue.toString(),
            maxValueLabel: maxValueLabel ?? maxValue.toString(),
            onMove: onMove,
            onRelease: onRelease
        };
        return slider;
    }
    export function showSlider(slider: Slider, parentDiv: Types.D3Selection): void {
        slider.base.div = parentDiv.append('div').attr('class', 'slider').attr('id', slider.base.id!) as any;
        let sliderLeftValue = slider.base.div.append('div').attr('class', 'slider-left-value').text(slider.minValueLabel) as any;
        let sliderRightValue = slider.base.div.append('div').attr('class', 'slider-right-value').text(slider.maxValueLabel) as any;
        let sliderMain = slider.base.div.append('div').attr('class', 'slider-main') as any;
        let sliderRailRuler = sliderMain.append('div').attr('class', 'slider-rail-ruler');
        sliderRailRuler.append('div').attr('class', 'slider-rail');
        let sliderRailClickable = sliderRailRuler.append('div').attr('class', 'slider-rail-clickable');
        let sliderHandleRuler = sliderRailRuler.append('div').attr('class', 'slider-handle-ruler');
        let sliderHandle = sliderHandleRuler.append('div').attr('class', 'slider-handle');
        let relativePosition = sliderValueToRelativePosition(slider, slider.selectedValue);
        sliderHandleRuler.style('left', relativePosition*100 + '%');
        Drawing.setTooltips(slider.base.viewer, sliderMain, [slider.base.tooltip], false, true);
        Drawing.setTooltips(slider.base.viewer, sliderLeftValue, ['Click/hold to decrease value.'], false, true);
        Drawing.setTooltips(slider.base.viewer, sliderRightValue, ['Click/hold to increase value.'], false, true);

        let dragHandler = d3.drag()
            .on('start.slider', () => moveSlider(slider, d3.event.sourceEvent))
            .on('drag.slider', () => moveSlider(slider, d3.event.sourceEvent))
            .on('end.slider', () => releaseSlider(slider, d3.event.sourceEvent));
        dragHandler(sliderRailClickable as any);
        dragHandler(sliderHandle as any);

        Drawing.addMouseHoldBehavior(sliderLeftValue as any, ()=>sliderStep(slider, -1), ()=>sliderStep(slider, -1), ()=>{});
        Drawing.addMouseHoldBehavior(sliderRightValue as any, ()=>sliderStep(slider, +1), ()=>sliderStep(slider, +1), ()=>{});

    }
    function sliderStep(slider: Slider, nSteps: number) {
        let value = Geometry.constrain(slider.selectedValue + nSteps*slider.step, slider.minValue, slider.maxValue);
        let relative = sliderValueToRelativePosition(slider, value);
        slider.base.div.select('div.slider-handle-ruler').style('left', relative*100 + '%');
        slider.selectedValue = value;
        slider.onMove(value);
        slider.onRelease(value);
    }

    function sliderValueToRelativePosition(slider: Slider, value: number): number {
        let relativePosition = (value - slider.minValue) / (slider.maxValue - slider.minValue);
        relativePosition = Geometry.constrain(relativePosition, 0, 1);
        return relativePosition;
    }
    function sliderRelativePositionToValue(slider: Slider, relativePosition: number): number {
        return slider.minValue + relativePosition * (slider.maxValue - slider.minValue);
    }
    function discretizeSliderValue(slider: Slider, value: number): number {
        let nSteps = Math.round((value - slider.minValue) / slider.step);
        return slider.minValue + nSteps*slider.step;
    }
    function getRelativePositionOnSlider(slider: Slider, event: MouseEvent): number|null {
        let ruler = slider.base.div.selectAll('div.slider-rail-ruler').node() as HTMLDivElement;
        if (ruler !== null){
            let {x, width } = ruler.getBoundingClientRect();
            let relativePosition = (event.clientX - x) / width;
            relativePosition = Geometry.constrain(relativePosition, 0, 1);
            return relativePosition;
        } else {
            return null;
        }
    }
    function moveSlider(slider: Slider, event: MouseEvent): void {
        let relativePosition = getRelativePositionOnSlider(slider, event);
        if (relativePosition !== null){
            let value = sliderRelativePositionToValue(slider, relativePosition);
            value = discretizeSliderValue(slider, value);
            relativePosition = sliderValueToRelativePosition(slider, value);
            slider.base.div.select('div.slider-handle-ruler').style('left', relativePosition*100 + '%');
            slider.onMove(value);
        }        
    }
    function releaseSlider(slider: Slider, event: MouseEvent): void {
        let relativePosition = getRelativePositionOnSlider(slider, event);
        if (relativePosition !== null){
            let value = sliderRelativePositionToValue(slider, relativePosition);
            value = discretizeSliderValue(slider, value);
            relativePosition = sliderValueToRelativePosition(slider, value);
            slider.base.div.select('div.slider-handle-ruler').style('left', relativePosition*100 + '%');
            slider.selectedValue = value;
            slider.onRelease(value);
        }
    }

    export function newPopupSlider(viewer: Types.Viewer, id: string|null, textPrefix: string, textSuffix: string, 
            minValue: number, maxValue: number, step: number, selectedValue: number, 
            minValueLabel: string|null, maxValueLabel: string|null, // if null, minValue and maxValue will be shown; if '', no label will be shown
            onMove: (value: number) => any, onRelease: (value: number) => any, tooltip: string|null, autocollapse: boolean = true): Popup {
        let popup = newPopup(viewer, id, textPrefix + selectedValue + textSuffix, autocollapse, tooltip);
        let slider = newSlider(viewer, id, minValue, maxValue, step, selectedValue, minValueLabel, maxValueLabel,
            value => { changePopupText(popup, textPrefix + value + textSuffix); onMove(value); },
            value => { changePopupText(popup, textPrefix + value + textSuffix); onRelease(value); },
            null
        );
        addToPopup(popup, slider);
        return popup;
    }

}