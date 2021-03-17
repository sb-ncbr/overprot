import * as d3 from 'd3';
import { Constants } from './Constants';
import { Geometry } from './Geometry';
export var Dag;
(function (Dag) {
    function newDag() {
        return { nodes: [], precedence: [], beta_connectivity: [], levels: [], precedenceLines: [], activeNodes: [], origPrecedence: [], origBetaConnectivity: [], error: null };
    }
    Dag.newDag = newDag;
    function dagFromJson(json) {
        var _a;
        let error = null;
        try {
            let dag = JSON.parse(json);
            error = dag.error = (_a = dag.error) !== null && _a !== void 0 ? _a : null;
            dag.nodes.forEach(node => node.active = true);
            dag.levels = [];
            dag.precedenceLines = [];
            dag.activeNodes = d3.range(dag.nodes.length);
            dag.origPrecedence = dag.precedence;
            dag.origBetaConnectivity = dag.beta_connectivity;
            return dag;
        }
        catch (ex) {
            error = error !== null && error !== void 0 ? error : `Failed to parse input file.`;
            console.warn(error, '\n', ex, '\n', json);
            return newDagWithError(error);
        }
    }
    Dag.dagFromJson = dagFromJson;
    function newDagWithError(error) {
        let dag = newDag();
        dag.error = error;
        return dag;
    }
    Dag.newDagWithError = newDagWithError;
    function newNodeVisual() {
        return { rect: Geometry.newRectangle(), fill: Constants.NODE_FILL.hex(), stroke: Constants.NODE_STROKE.hex() };
    }
    Dag.newNodeVisual = newNodeVisual;
    function filterDagAndAddLevels(original, occurrence_threshold) {
        if (occurrence_threshold == 0.0) {
            original.nodes.forEach(node => node.active = true);
            original.activeNodes = d3.range(original.nodes.length);
            original.precedence = original.origPrecedence;
            original.beta_connectivity = original.origBetaConnectivity;
            addLevels(original);
            // return original;
        }
        else {
            let nNodes = original.nodes.length;
            original.nodes.forEach(node => node.active = node.occurrence >= occurrence_threshold);
            let selectedIndices = d3.range(nNodes).filter(i => original.nodes[i].active);
            // let selectedIndices = d3.range(nNodes).filter(i => original.nodes[i].occurrence >= occurrence_threshold);
            // let indexMap = new Array(nNodes).fill(-1);
            // selectedIndices.forEach((iOld, iNew) => indexMap[iOld] = iNew);
            let { ins: before, outs: after } = getInAndOutNeighbors(nNodes, original.origPrecedence);
            // let { ins: before, outs: after } = getInAndOutNeighbors(nNodes, original.precedence);
            for (let i = 0; i < nNodes; i++) {
                if (!original.nodes[i].active) { // removing node
                    // if (indexMap[i] < 0) {  // removing node
                    before[i].forEach(b => {
                        after[i].forEach(a => {
                            if (!(after[b].includes(a))) {
                                after[b].push(a);
                                before[a].push(b);
                            }
                        });
                    });
                }
            }
            // let result = Dag.newDag();
            // result.nodes = original.nodes;// selectedIndices.map(i => original.nodes[i]);
            original.activeNodes = selectedIndices;
            // result.activeNodes = selectedIndices;
            original.precedence = [];
            selectedIndices.forEach(b => {
                after[b].forEach(a => {
                    if (original.nodes[a].active)
                        original.precedence.push([b, a]);
                });
            });
            // result.precedence = [];
            // selectedIndices.forEach(b => {
            //     after[b].forEach(a => {
            //         if (indexMap[a] >= 0)
            //             result.precedence.push([indexMap[b], indexMap[a]]);
            //     });
            // });
            addLevels(original);
            removeRedundantPrecedenceEdges(original);
            // return original;
            // addLevels(result);
            // removeRedundantPrecedenceEdges(result);
            // console.log('precedence:', result.precedence);
            // return result;
        }
    }
    Dag.filterDagAndAddLevels = filterDagAndAddLevels;
    function addLevels(dag) {
        let levels = [];
        let todoNodes = new Set(dag.activeNodes);
        // let todoNodes = new Set(d3.range(dag.nodes.length));
        let todoEdges = dag.precedence;
        while (todoNodes.size > 0) {
            let minVertices = new Set(todoNodes);
            todoEdges.forEach(e => minVertices.delete(e[1])); //remove vertices which are preceded by another vertex
            todoEdges = todoEdges.filter(e => !minVertices.has(e[0]));
            minVertices.forEach(v => todoNodes.delete(v));
            levels.push(Array.from(minVertices));
        }
        dag.levels = levels;
    }
    function removeRedundantPrecedenceEdges(dag) {
        let nNodes = dag.nodes.length;
        let nodeLevel = new Array(nNodes).fill(-1); // -1 = inactive node
        dag.levels.forEach((level, iLevel) => level.forEach(iNode => nodeLevel[iNode] = iLevel));
        let newPrecedence = [];
        let { outs: after } = getInAndOutNeighbors(nNodes, dag.precedence);
        for (let iLevel = 0; iLevel < dag.levels.length; iLevel++) {
            const level = dag.levels[iLevel];
            level.forEach(iNode => {
                after[iNode].forEach(a => {
                    if (nodeLevel[a] == iLevel + 1 || !existsTransitivePath(iNode, a, after, nodeLevel)) {
                        newPrecedence.push([iNode, a]);
                    }
                });
            });
        }
        dag.precedence = newPrecedence;
    }
    function existsTransitivePath(iFrom, iTo, outNeighbors, nodeLevel) {
        // Finding path of length > 1 from iFrom to iTo by DFS:
        let todo = outNeighbors[iFrom].filter(v => v != iTo);
        while (todo.length > 0) {
            let v = todo.pop();
            if (nodeLevel[v] != -1 && nodeLevel[v] < nodeLevel[iTo]) { // -1 = inactive node
                todo.push(...outNeighbors[v]);
            }
            else if (v == iTo) {
                return true;
            }
        }
        return false;
    }
    function getInAndOutNeighbors(nNodes, edges) {
        let ins = new Array(nNodes);
        let outs = new Array(nNodes);
        for (let i = 0; i < nNodes; i++) {
            ins[i] = [];
            outs[i] = [];
        }
        edges.forEach(edge => {
            ins[edge[1]].push(edge[0]);
            outs[edge[0]].push(edge[1]);
        });
        return { ins: ins, outs: outs };
    }
    function getNodeMinMaxLength(node) {
        let lengthLevels = node.cdf.map(lp => lp[0]).filter(l => l > 0);
        if (lengthLevels.length == 0) {
            return [0, 0];
        }
        else {
            return [lengthLevels[0], lengthLevels[lengthLevels.length - 1]];
        }
    }
    Dag.getNodeMinMaxLength = getNodeMinMaxLength;
})(Dag || (Dag = {}));
//# sourceMappingURL=Dag.js.map