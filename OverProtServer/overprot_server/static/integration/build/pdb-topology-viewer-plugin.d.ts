declare function measureExecTime(foo: any, args: any): void;
declare function getStartStopCoords(d3selection: any): {
    startCoords: {
        x: undefined;
        y: undefined;
    };
    stopCoords: {
        x: undefined;
        y: undefined;
    };
};
declare function getPathPointAfterTransform(xcoord: any, ycoord: any, pathEle: any): {
    x: any;
    y: any;
};
declare function applyRotationMatrix(point: any, center: any, angle: any): {
    x: any;
    y: any;
};
declare function composePathHelix(center: any, MINORAXIS: any, sse: any, CONVEXITY: any): any[];
declare function composePathStrand(center: any, MINORAXIS: any, sse: any, ARROW_HEIGHT: any, ARROW_SPREAD: any): any[];
declare function convertPathCartesianToYReversed(pathCartesian: any, lowerLeft: any, upperRight: any): any;
declare function convert2DProtsJSONtoTopologyAPIJSON(inputJson: any, entryID: any, entityID: any, chainID: any): {} | undefined;
declare class PdbTopologyViewerPlugin {
    defaultColours: {
        domainSelection: string;
        mouseOver: string;
        borderColor: string;
        qualityGreen: string;
        qualityRed: string;
        qualityYellow: string;
        qualityOrange: string;
    };
    displayStyle: string;
    errorStyle: string;
    menuStyle: string;
    familyId: string;
    domainId: string;
    structAsymId: string;
    twoDProtsTimestamp: string;
    sequenceArr: string[];
    entityId: string;
    entryId: string;
    chainId: string;
    apiData: any;
    twoDProtsData: {
        topologyData: undefined;
        residueNumbers: undefined;
    };
    targetEle: HTMLElement;
    pdbevents: any;
    xScale: any;
    yScale: any;
    zoom: any;
    scaledPointsArr: any[];
    domainTypes: any[];
    svgWidth: number;
    svgHeight: number;
    svgEle: any;
    subscribeEvents: boolean;
    render(target: HTMLElement, options: {
        domainId: string;
        familyId: string;
        entityId: string;
        entryId: string;
        chainId?: string;
        structAsymId?: string;
        twoDProtsTimestamp?: string;
        subscribeEvents?: boolean;
        displayStyle?: string;
        errorStyle?: string;
        menuStyle?: string;
    }): void;
    initPainting(): void;
    displayError(errType?: string): void;
    createNewEvent: (eventTypeArr: string[]) => any;
    getObservedResidues(pdbId: string): Promise<any>;
    getApiData(pdbId: string, entityId: string, chainId: string, familyId: string, domainId: string, structAsymId: string, twoDProtsTimestamp: string): Promise<any[]>;
    getPDBSequenceArray(entities: any[]): void;
    chunkArray(arr: any[], len: number): any[][];
    getDomainRange(): void;
    drawStrandSubpaths(startResidueNumber: number, stopResidueNumber: number, index: number, parentSSEId: string): void;
    drawStrandMaskShape(index: number): void;
    renderTooltip(elementData: any, action: string): void;
    dispatchEvent(eventType: any, eventData: any, eventElement?: HTMLElement): void;
    clickAction(eleObj: any): void;
    mouseoverAction(eleObj: any | this, eleData: any): void;
    mouseoutAction(eleObj: any, eleData: any): void;
    drawHelicesSubpaths(startResidueNumber: number, stopResidueNumber: number, index: number, curveYdiff: number, parentSSEId: string): void;
    drawHelicesMaskShape(index: number): void;
    drawCoilsSubpaths(startResidueNumber: number, stopResidueNumber: number, index: number, color: string): void;
    drawConnectingCoils(): void;
    drawTopologyStructures(): void;
    zoomDraw(): void;
    clearHighlight(): void;
    highlight(startResidue: number, endResidue: number, color?: {
        r: number;
        g: number;
        b: number;
    } | string, eventType?: string): void;
    drawValidationShape(residueNumber: number, shape: string, rgbColor: string): void;
    getAnnotationFromMappings: () => void;
    getChainStartAndEnd(): {
        start: number;
        end: number;
    } | undefined;
    getAnnotationFromOutliers(): void;
    createDomainDropdown: () => void;
    resetTheme(): void;
    changeResidueColor(residueNumber: number, rgbColor: string, tooltipContent: string, tooltipPosition: string): void;
    updateTheme(residueDetails: any): void;
    displayDomain(invokedFrom?: string): void;
    resetDisplay(): void;
    handleSeqViewerEvents(e: any, eType: string): void;
    handleProtvistaEvents(e: any, eType: string): void;
    handleMolstarEvents(e: any, eType: string): void;
    subscribeWcEvents(): void;
}
