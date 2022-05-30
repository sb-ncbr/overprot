import * as d3 from 'd3';
import { Constants } from './Constants';
import { Drawing } from './Drawing';
import { Geometry } from './Geometry';
export var Controls;
(function (Controls) {
    function emptySelection() {
        return d3.selectAll();
    }
    function newControlPanel(viewer, id, tooltip) {
        let panel = {
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showControlPanel(panel, par) }
        };
        return panel;
    }
    Controls.newControlPanel = newControlPanel;
    function addToControlPanel(panel, child) {
        panel.base.children.push(child);
    }
    Controls.addToControlPanel = addToControlPanel;
    function showControlPanel(panel, parentDiv) {
        panel.base.div = parentDiv.append('div').attr('class', 'control-panel').attr('id', panel.base.id);
        panel.base.children.forEach(child => child.base.show(panel.base.div));
    }
    function newButton(viewer, id, text, square, icon, onClick, tooltip) {
        let button = {
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showButton(button, par) },
            text: text,
            square: square,
            icon: icon,
            onClick: onClick
        };
        return button;
    }
    Controls.newButton = newButton;
    function showButton(button, parentDiv) {
        button.base.div = parentDiv.append('div').attr('class', button.square ? 'button square-button' : 'button').attr('id', button.base.id);
        const clas = button.icon ? 'button-icon' : 'button-text';
        button.base.div.append('div').attr('class', clas).html(button.text);
        Drawing.setTooltips(button.base.viewer, button.base.div, [button.base.tooltip], false, true);
        button.base.div.on('click', button.onClick);
        button.base.div.on('dblclick', () => { d3.event.stopPropagation(); });
    }
    function changeButtonText(button, newText) {
        button.text = newText;
        button.base.div.select('div.button-icon,div.button-text').html(newText);
    }
    Controls.changeButtonText = changeButtonText;
    function newToggleButton(viewer, id, text, square, icon, selectedValue, onSelect, tooltip) {
        let isChecked = selectedValue;
        if (text instanceof String) {
        }
        let textOff = typeof text == 'string' ? '<span style="opacity:0;">&check;</span>' + text : text[0];
        let textOn = typeof text == 'string' ? '&check;' + text : text[1];
        let button = {
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showButton(button, par) },
            text: isChecked ? textOn : textOff,
            square: square,
            icon: icon,
            onClick: () => {
                isChecked = !isChecked;
                onSelect(isChecked);
                changeButtonText(button, isChecked ? textOn : textOff);
            }
        };
        return button;
    }
    Controls.newToggleButton = newToggleButton;
    function updateToggleButtonText(button, isChecked, text) {
        changeButtonText(button, isChecked ? '&check;' + text : '<span style="opacity:0;">&check;</span>' + text);
    }
    function newPopup(viewer, id, text, autocollapse, tooltip) {
        // console.log('newPopup');
        let popup = {
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showPopup(popup, par) },
            headButton: newButton(viewer, null, text + Constants.OPEN_POPUP_SYMBOL, false, false, () => togglePopup(popup), tooltip),
            autocollapse: autocollapse
        };
        return popup;
    }
    Controls.newPopup = newPopup;
    function addToPopup(popup, child) {
        // console.log('addToPopup', popup, child);
        popup.base.children.push(child);
    }
    Controls.addToPopup = addToPopup;
    function showPopup(popup, parentDiv) {
        popup.base.div = parentDiv.append('div').attr('class', popup.autocollapse ? 'popup autocollapse' : 'popup').attr('id', popup.base.id);
        let headDiv = popup.base.div.append('div').attr('class', 'popup-head');
        popup.base.div.data([popup]);
        showButton(popup.headButton, headDiv);
    }
    function togglePopup(popup) {
        // console.log('togglePopup', popup);
        let expanded = popup.base.div.selectAll('div.popup-tail').size() > 0;
        if (expanded) {
            collapsePopup(popup);
        }
        else {
            collapseAllAutocollapsePopups(popup.base.viewer);
            expandPopup(popup);
            d3.event.stopPropagation();
        }
    }
    function expandPopup(popup) {
        // console.log('expandPopup');
        let headDiv = popup.base.div.select('div.popup-head');
        let tailDiv = popup.base.div.append('div').attr('class', 'popup-tail');
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
    function collapsePopup(popup) {
        // console.log('collapsePopup');
        popup.base.div.selectAll('div.popup-tail').remove();
    }
    function collapseAllAutocollapsePopups(viewer) {
        viewer.mainDiv.selectAll('div.popup.autocollapse').each(d => collapsePopup(d));
    }
    function changePopupText(popup, newText) {
        changeButtonText(popup.headButton, newText + Constants.OPEN_POPUP_SYMBOL);
    }
    function newListbox(viewer, id, namesValuesTooltips, selectedValue, onSelect, tooltip) {
        let listbox = {
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showListbox(listbox, par) },
            namesValuesTooltips: namesValuesTooltips,
            selectedValue: selectedValue,
            onSelect: onSelect
        };
        return listbox;
    }
    Controls.newListbox = newListbox;
    function showListbox(listbox, parentDiv) {
        listbox.base.div = parentDiv.append('div').attr('class', 'listbox').attr('id', listbox.base.id);
        let itemOnClick = (nvt) => {
            listbox.selectedValue = nvt[1];
            listbox.base.div.selectAll('div.listbox-item')
                .attr('class', nvt2 => nvt2[1] == listbox.selectedValue ? 'button listbox-item selected' : 'button listbox-item');
            listbox.onSelect(nvt[1]);
        };
        listbox.base.div.selectAll().data(listbox.namesValuesTooltips)
            .enter().append('div').attr('class', kv => kv[1] == listbox.selectedValue ? 'button listbox-item selected' : 'button listbox-item')
            .on('click', itemOnClick)
            .append('div').attr('class', 'button-text').html(kv => kv[0]);
        Drawing.setTooltips(listbox.base.viewer, listbox.base.div.selectAll('div.listbox-item'), listbox.namesValuesTooltips.map(nvt => nvt[2]), false, true);
        Drawing.setTooltips(listbox.base.viewer, listbox.base.div, [listbox.base.tooltip], false, true);
    }
    function newDropdownList(viewer, id, text, namesValuesTooltips, selectedValue, onSelect, tooltip, autocollapse = true, collapseOnSelect = true) {
        let popup = newPopup(viewer, id, text, autocollapse, tooltip);
        let wrappedOnSelect = collapseOnSelect ?
            (value) => { collapsePopup(popup); onSelect(value); }
            : onSelect;
        let listbox = newListbox(viewer, id, namesValuesTooltips, selectedValue, wrappedOnSelect, null);
        addToPopup(popup, listbox);
        return popup;
    }
    Controls.newDropdownList = newDropdownList;
    function newSlider(viewer, id, minValue, maxValue, step, selectedValue, minValueLabel, maxValueLabel, // if null, minValue and maxValue will be shown; if '', no label will be shown
    onMove, onRelease, tooltip) {
        // console.log('newSlider');
        let slider = {
            base: { viewer: viewer, id: id, children: [], div: emptySelection(), tooltip: tooltip, show: (par) => showSlider(slider, par) },
            minValue: minValue,
            maxValue: maxValue,
            step: step,
            selectedValue: selectedValue,
            minValueLabel: minValueLabel !== null && minValueLabel !== void 0 ? minValueLabel : minValue.toString(),
            maxValueLabel: maxValueLabel !== null && maxValueLabel !== void 0 ? maxValueLabel : maxValue.toString(),
            onMove: onMove,
            onRelease: onRelease
        };
        return slider;
    }
    Controls.newSlider = newSlider;
    function showSlider(slider, parentDiv) {
        slider.base.div = parentDiv.append('div').attr('class', 'slider').attr('id', slider.base.id);
        let sliderLeftValue = slider.base.div.append('div').attr('class', 'slider-left-value').text(slider.minValueLabel);
        let sliderRightValue = slider.base.div.append('div').attr('class', 'slider-right-value').text(slider.maxValueLabel);
        let sliderMain = slider.base.div.append('div').attr('class', 'slider-main');
        let sliderRailRuler = sliderMain.append('div').attr('class', 'slider-rail-ruler');
        sliderRailRuler.append('div').attr('class', 'slider-rail');
        let sliderRailClickable = sliderRailRuler.append('div').attr('class', 'slider-rail-clickable');
        let sliderHandleRuler = sliderRailRuler.append('div').attr('class', 'slider-handle-ruler');
        let sliderHandle = sliderHandleRuler.append('div').attr('class', 'slider-handle');
        let relativePosition = sliderValueToRelativePosition(slider, slider.selectedValue);
        sliderHandleRuler.style('left', relativePosition * 100 + '%');
        Drawing.setTooltips(slider.base.viewer, sliderMain, [slider.base.tooltip], false, true);
        Drawing.setTooltips(slider.base.viewer, sliderLeftValue, ['Click/hold to decrease value.'], false, true);
        Drawing.setTooltips(slider.base.viewer, sliderRightValue, ['Click/hold to increase value.'], false, true);
        let dragHandler = d3.drag()
            .on('start.slider', () => moveSlider(slider, d3.event.sourceEvent))
            .on('drag.slider', () => moveSlider(slider, d3.event.sourceEvent))
            .on('end.slider', () => releaseSlider(slider, d3.event.sourceEvent));
        dragHandler(sliderRailClickable);
        dragHandler(sliderHandle);
        Drawing.addMouseHoldBehavior(sliderLeftValue, () => sliderStep(slider, -1), () => sliderStep(slider, -1), () => { });
        Drawing.addMouseHoldBehavior(sliderRightValue, () => sliderStep(slider, +1), () => sliderStep(slider, +1), () => { });
    }
    Controls.showSlider = showSlider;
    function sliderStep(slider, nSteps) {
        let value = Geometry.constrain(slider.selectedValue + nSteps * slider.step, slider.minValue, slider.maxValue);
        let relative = sliderValueToRelativePosition(slider, value);
        slider.base.div.select('div.slider-handle-ruler').style('left', relative * 100 + '%');
        slider.selectedValue = value;
        slider.onMove(value);
        slider.onRelease(value);
    }
    function sliderValueToRelativePosition(slider, value) {
        let relativePosition = (value - slider.minValue) / (slider.maxValue - slider.minValue);
        relativePosition = Geometry.constrain(relativePosition, 0, 1);
        return relativePosition;
    }
    function sliderRelativePositionToValue(slider, relativePosition) {
        return slider.minValue + relativePosition * (slider.maxValue - slider.minValue);
    }
    function discretizeSliderValue(slider, value) {
        let nSteps = Math.round((value - slider.minValue) / slider.step);
        return slider.minValue + nSteps * slider.step;
    }
    function getRelativePositionOnSlider(slider, event) {
        let ruler = slider.base.div.selectAll('div.slider-rail-ruler').node();
        if (ruler !== null) {
            let { x, width } = ruler.getBoundingClientRect();
            let relativePosition = (event.clientX - x) / width;
            relativePosition = Geometry.constrain(relativePosition, 0, 1);
            return relativePosition;
        }
        else {
            return null;
        }
    }
    function moveSlider(slider, event) {
        let relativePosition = getRelativePositionOnSlider(slider, event);
        if (relativePosition !== null) {
            let value = sliderRelativePositionToValue(slider, relativePosition);
            value = discretizeSliderValue(slider, value);
            relativePosition = sliderValueToRelativePosition(slider, value);
            slider.base.div.select('div.slider-handle-ruler').style('left', relativePosition * 100 + '%');
            slider.onMove(value);
        }
    }
    function releaseSlider(slider, event) {
        let relativePosition = getRelativePositionOnSlider(slider, event);
        if (relativePosition !== null) {
            let value = sliderRelativePositionToValue(slider, relativePosition);
            value = discretizeSliderValue(slider, value);
            relativePosition = sliderValueToRelativePosition(slider, value);
            slider.base.div.select('div.slider-handle-ruler').style('left', relativePosition * 100 + '%');
            slider.selectedValue = value;
            slider.onRelease(value);
        }
    }
    function newPopupSlider(viewer, id, textPrefix, textSuffix, minValue, maxValue, step, selectedValue, minValueLabel, maxValueLabel, // if null, minValue and maxValue will be shown; if '', no label will be shown
    onMove, onRelease, tooltip, autocollapse = true) {
        let popup = newPopup(viewer, id, textPrefix + selectedValue + textSuffix, autocollapse, tooltip);
        let slider = newSlider(viewer, id, minValue, maxValue, step, selectedValue, minValueLabel, maxValueLabel, value => { changePopupText(popup, textPrefix + value + textSuffix); onMove(value); }, value => { changePopupText(popup, textPrefix + value + textSuffix); onRelease(value); }, null);
        addToPopup(popup, slider);
        return popup;
    }
    Controls.newPopupSlider = newPopupSlider;
})(Controls || (Controls = {}));
//# sourceMappingURL=Controls.js.map