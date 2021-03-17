import { OverProtViewerCore as OverProtViewerCore } from './OverProtViewerCore';
var OverProtViewer;
(function (OverProtViewer) {
    class OverProtViewerElement extends HTMLElement {
        constructor() {
            super();
        }
        connectedCallback() {
            this.initialize();
        }
        /** Initialize this overprot-viewer element using its HTML attributes.
         * Return true on success, false on failure to load data.
        */
        initialize() {
            return OverProtViewerCore.initializeViewer(this);
        }
    }
    window.customElements.define('overprot-viewer', OverProtViewerElement);
})(OverProtViewer || (OverProtViewer = {}));
//# sourceMappingURL=OverProtViewer.js.map