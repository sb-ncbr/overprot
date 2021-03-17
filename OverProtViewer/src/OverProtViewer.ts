import { OverProtViewerCore as OverProtViewerCore } from './OverProtViewerCore';


namespace OverProtViewer {

    class OverProtViewerElement extends HTMLElement {
        public constructor() {
            super();
        }
        public connectedCallback(): void {
            this.initialize();
        }
        /** Initialize this overprot-viewer element using its HTML attributes. 
         * Return true on success, false on failure to load data.
        */
        public initialize(): void {
            return OverProtViewerCore.initializeViewer(this);
        }
    }

    window.customElements.define('overprot-viewer', OverProtViewerElement);
}