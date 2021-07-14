
export namespace Graphs {

    type XY = { x: number, y: number };

    function addXY(xy1: XY, xy2: XY): XY {
        return { x: xy1.x + xy2.x, y: xy1.y + xy2.y };
    }

    type Size = { width: number, height: number, weight: number };

    type Box = { left: number, right: number, top: number, bottom: number, weight: number }
    /* Boundaries of a box with center in [0,0]. All dimensions are positive. Center means geometrical center along x, center of gravity along y. */

    export type Dag = {
        levels: number[][],
        edges: [number, number][],
        vertices: number[],
        in_neighbors: Map<number, number[]>
    }

    export function newDagEmpty(): Dag {
        return { levels: [], edges: [], vertices: [], in_neighbors: new Map() };
    }
    export function newDagFromPath(vertices: number[]): Dag {
        let dag: Dag = { levels: vertices.map(v => [v]), edges: [], vertices: [...vertices], in_neighbors: new Map() };
        for (let i = 0; i < vertices.length - 1; i++) {
            dag.edges.push([vertices[i], vertices[i + 1]]);
        }
        addInNeighbors(dag);
        return dag;
    }
    export function newDagFromPrecedence(levels: number[][], edges: [number, number][]): Dag {
        let vertices: number[] = [];
        for (const level of levels) for (const v of level) vertices.push(v);
        let vertexSet = new Set(vertices);
        let finalEdges: [number, number][] = [];
        for (const e of edges) if (vertexSet.has(e[0]) && vertexSet.has(e[1])) finalEdges.push(e);
        let dag: Dag = { levels: levels, edges: finalEdges, vertices: vertices, in_neighbors: new Map() };
        addInNeighbors(dag);
        return dag;
    }

    function addInNeighbors(dag: Dag): void {
        for (const v of dag.vertices) dag.in_neighbors.set(v, []);
        for (const [u, v] of dag.edges) dag.in_neighbors.get(v)!.push(u);
    }

    function getSlice(dag: Dag, levelFrom: number, levelTo: number): Dag {
        let sliceLevels = dag.levels.slice(levelFrom, levelTo);
        let sliceEdges: [number, number][] = [];
        for (let iLevel = 1; iLevel < sliceLevels.length; iLevel++) {
            for (const v of sliceLevels[iLevel]) {
                for (const u of dag.in_neighbors.get(v)!) {
                    sliceEdges.push([u, v]);
                }
            }
        }
        let result = newDagFromPrecedence(sliceLevels, sliceEdges);
        // Debug check:
        // for (const [u, v] of result.edges) {
        //     assert(result.vertices.includes(u), u, 'not in vertices');
        //     assert(result.vertices.includes(v), v, 'not in vertices');
        // }
        return result;
    }

    function getSlices(dag: Dag): Dag[] {
        /* Find all n "slice-vertices" v[i] such that 
            Vertex v is a slice-vertex <==> foreach vertex u. exists path u->v or exists path v-> or u==v
        Find all n+1 "slices" S[i] such that
            u in S[0] <==> exists path         u->v[0]
            u in S[i] <==> exists path v[i-1]->u->v[i]
            u in S[n] <==> exists path v[n-1]->u
        Return list S[0], v[0], S[1], v[1], ... v[n-1], S[n]. (ommitting empty S[i]) */
        let ancestors = new Map<number, Set<number>>();
        let seenVertices = 0;
        let slices = [];
        let lastSliceVertex = -1;
        const nLevels = dag.levels.length;
        for (let iLevel = 0; iLevel < nLevels; iLevel++) {
            const level = dag.levels[iLevel];
            for (let iVertex = 0; iVertex < level.length; iVertex++) {
                const v = level[iVertex];
                const ancestors_v = new Set(dag.in_neighbors.get(v));
                for (const u of dag.in_neighbors.get(v)!) for (const t of ancestors.get(u)!) ancestors_v.add(t);
                ancestors.set(v, ancestors_v);
            }
            if (level.length == 1 && ancestors.get(level[0])!.size == seenVertices) {
                // This is a slice-vertex
                const sliceVertex = level[0];
                if (iLevel > lastSliceVertex + 1) {
                    slices.push(getSlice(dag, lastSliceVertex + 1, iLevel));
                }
                slices.push(newDagFromPrecedence([[sliceVertex]], []));
                ancestors.clear();
                ancestors.set(sliceVertex, new Set());
                lastSliceVertex = iLevel;
                seenVertices = 1;
            } else {
                seenVertices += level.length;
            }
        }
        if (nLevels > lastSliceVertex + 1) {
            slices.push(getSlice(dag, lastSliceVertex + 1, nLevels));
        }
        // Debug check:
        // let checkVertices = new Set();
        // for (const slice of slices) for (const v of slice.vertices) checkVertices.add(v);
        // assert(setsEqual(checkVertices, new Set(dag.vertices)), 'checkVertices', checkVertices, '!= dag.vertices', dag.vertices);
        return slices;
    }

    function setsEqual<T>(a: Set<T>, b: Set<T>): boolean {
        if (a.size != b.size) return false;
        for (const x of a) {
            if (!b.has(x)) return false;
        }
        return true;
    }

    export function embedDag(dag: Dag, vertexSizes: Map<number, Size>, padding: XY = { x: 0.0, y: 0.0 },
        leftMargin: number = 0.0, rightMargin: number = 0.0, topMargin: number = 0.0, bottomMargin: number = 0.0): [Box, Map<number, XY>] {
        /* Place vertices in a plane so that all edges go from left to right, use heuristics to make it look nice.
        Return the bounding box and positions of individual vertices. */
        let box: Box;
        let positions: Map<number, XY>;
        if (dag.vertices.length == 0) {
            box = { left: 0.0, right: 0.0, top: 0.0, bottom: 0.0, weight: 0.0 };
            positions = new Map();
        } else {
            [box, positions] = embedGeneralDag(dag, vertexSizes, padding);
        }
        box.left += leftMargin;
        box.right += rightMargin;
        box.top += topMargin;
        box.bottom += bottomMargin;
        return [box, positions];
    }

    function embedGeneralDag(dag: Dag, vertex_sizes: Map<number, Size>, padding: XY): [Box, Map<number, XY>] {
        /* Embed non-empty DAG with possibly more than one component. */
        let components = dagConnectedComponents(dag);
        let embeddings = components.map(comp => embedConnectedDag(comp, vertex_sizes, padding));
        let grandEmbedding = combineEmbeddings(embeddings, Direction.VERTICAL_REARRANGED, padding);
        return grandEmbedding;
    }

    function embedConnectedDag(dag: Dag, vertexSizes: Map<number, Size>, padding: XY): [Box, Map<number, XY>] {
        /* Embed non-empty DAG with exactly one component. */
        const slices = getSlices(dag);
        if (slices.length == 1) {
            return embedUnsliceableDag(slices[0], vertexSizes, padding);
        } else {
            const embeddings = slices.map(s => embedGeneralDag(s, vertexSizes, padding));
            const grandEmbedding = combineEmbeddings(embeddings, Direction.HORIZONTAL, padding);
            return grandEmbedding;
        }
    }

    function embedUnsliceableDag(dag: Dag, vertexSizes: Map<number, Size>, padding: XY, maxAllowedPermutations: number = 1024): [Box, Map<number, XY>] {
        /* Embed non-empty DAG with exactly one component, without trying to slice it. */
        const nPerm = permutationNumber(dag.levels, maxAllowedPermutations + 1);
        if (dag.vertices.length == 1 || nPerm > maxAllowedPermutations) {
            // Do not permutate (not needed or too expensive)
            return embedLevels(dag.levels, vertexSizes, padding = padding);
        } else {
            // Permutate all levels to find the best tuple of permutations
            let bestEmbedding = null;
            let bestPenalty = Infinity;
            const levelPermutations = dag.levels.map(level => permutations(level, true));
            for (const perm of cartesianProduct(levelPermutations)) {
                const embedding = embedLevels(perm, vertexSizes, padding);
                const [box, positions] = embedding;
                let penalty = 0.0;
                for (const [u, v] of dag.edges) penalty += Math.abs(positions.get(u)!.y - positions.get(v)!.y);
                if (penalty < bestPenalty) {
                    bestEmbedding = embedding;
                    bestPenalty = penalty;
                }
            }
            assert(bestEmbedding != null, 'bestEmbedding is null', bestEmbedding);
            return bestEmbedding as [Box, Map<number, XY>];
        }
    }

    function embedLevels(levels: number[][], vertexSizes: Map<number, Size>, padding: XY): [Box, Map<number, XY>] {
        const levelEmbeddings = [];
        for (const level of levels) {
            const vertexEmbeddings: [Box, Map<number, XY>][] = [];
            for (const vertex of level) {
                const { width, height, weight } = vertexSizes.get(vertex)!;
                const box: Box = { left: width / 2, right: width / 2, top: height / 2, bottom: height / 2, weight: weight };
                const positions = new Map([[vertex, { x: 0.0, y: 0.0 } as XY]]);
                vertexEmbeddings.push([box, positions]);
            }
            const levelEmbedding = combineEmbeddings(vertexEmbeddings, Direction.VERTICAL, padding);
            levelEmbeddings.push(levelEmbedding);
        }
        const grandEmbedding = combineEmbeddings(levelEmbeddings, Direction.HORIZONTAL, padding);
        return grandEmbedding;
    }

    function permutationNumber(levels: number[][], maxAllowed: number) {
        /* Return the cardinality of the cartesian product of permutations of each level, 
        or return maxAllowed if it becomes clear that the result would be > maxAllowed. */
        let result = 1;
        for (const level of levels) {
            for (let i = level.length; i > 0; i--) {
                result *= i;
                if (result > maxAllowed) return maxAllowed;
            }
        }
        return result;
    }

    function permutations<T>(elements: T[], reversed: boolean = false): T[][] {
        if (elements.length == 0) return [[]];
        const result = [];
        let head = elements[0];
        let rest = elements.slice(1, undefined);
        for (let i = 0; i < elements.length; i++) {
            for (const subperm of permutations(rest, true)) {
                subperm.push(head);
                result.push(subperm);
            }
            [head, rest[i]] = [rest[i], head];
        }
        if (!reversed) for (const perm of result) perm.reverse();
        return result;
    }

    function cartesianProduct<T>(sets: T[][], skip: number = 0): T[][] {
        if (skip == sets.length) return [[]];
        const result = [];
        for (const head of sets[skip]) {
            for (const tuples of cartesianProduct(sets, skip + 1)) {
                tuples.push(head);
                result.push(tuples);
            }
        }
        if (skip == 0) for (const tup of result) tup.reverse();
        return result;
    }

    enum Direction { HORIZONTAL, VERTICAL, VERTICAL_REARRANGED };

    function combineEmbeddings(embeddings: [Box, Map<number, XY>][], direction: Direction, padding: XY): [Box, Map<number, XY>] {
        const boxes = embeddings.map(emb => emb[0]);
        const positions = embeddings.map(emb => emb[1]);
        let grandBox: Box;
        let boxPlaces: XY[];
        if (direction == Direction.HORIZONTAL) [grandBox, boxPlaces] = stackBoxesHorizontally(boxes, padding);
        else if (direction == Direction.VERTICAL) [grandBox, boxPlaces] = stackBoxesVertically(boxes, padding);
        else if (direction == Direction.VERTICAL_REARRANGED) [grandBox, boxPlaces] = stackBoxesVertically(boxes, padding, true);
        else throw new Error("direction must be HORIZONTAL or VERTICAL or VERTICAL_REARRANGED");
        let grandPositions = new Map<number, XY>();
        for (let i = 0; i < embeddings.length; i++) {
            for (const [vertex, xy] of positions[i]) {
                grandPositions.set(vertex, addXY(boxPlaces[i], xy));
            }
        }
        return [grandBox, grandPositions];
    }

    function sum(array: number[]): number {
        let result = 0;
        for (const x of array) result += x;
        return result;
    }

    function stackBoxesHorizontally(boxes: Box[], padding: XY): [Box, XY[]] {
        /* Return the grand box containing all boxes, and the list of positions of boxes in the grand box. */
        const grandWidth = sum(boxes.map(box => box.left + box.right)) + (boxes.length - 1) * padding.x
        const grandLeft = grandWidth / 2;
        const grandRight = grandLeft;
        const grandTop = Math.max(...boxes.map(box => box.top));
        const grandBottom = Math.max(...boxes.map(box => box.bottom));
        const grandWeight = sum(boxes.map(box => box.weight));
        let boxPlaces: XY[] = [];
        let x = -grandLeft;
        for (const box of boxes) {
            x += box.left;
            boxPlaces.push({ x: x, y: 0.0 });
            x += box.right;
            x += padding.x;
        }
        const grandBox: Box = { left: grandLeft, right: grandRight, top: grandTop, bottom: grandBottom, weight: grandWeight };
        return [grandBox, boxPlaces];
    }

    function stackBoxesVertically(boxes: Box[], padding: XY, rearrange: boolean = false): [Box, XY[]] {
        /* Return the grand box containing all boxes, and the list of positions of boxes in the grand box. */
        const nBoxes = boxes.length;
        let sumW = 0.0;
        let sumWY = 0.0;
        let y = 0.0;
        const rearrangedBoxes: [number, Box][] = rearrange ? rearrangeBoxesFromMiddle(boxes) : boxes.map((box, i) => [i, box]);
        for (const [i, box] of rearrangedBoxes) {
            y += box.top;
            sumW += box.weight;
            sumWY += box.weight * y;
            y += box.bottom;
            y += padding.y;
        }
        const meanY = sumW > 0 ? sumWY / sumW : 0;
        const boxPlaces: XY[] = [];
        y = -meanY;
        for (const [i, box] of rearrangedBoxes) {
            y += box.top;
            boxPlaces[i] = { x: 0.0, y: y };
            y += box.bottom;
            y += padding.y;
        }
        const grandTop = meanY;
        const grandBottom = y - padding.y;
        const grandLeft = Math.max(...boxes.map(box => box.left));
        const grandRight = Math.max(...boxes.map(box => box.right));
        const grandBox: Box = { left: grandLeft, right: grandRight, top: grandTop, bottom: grandBottom, weight: sumW };
        return [grandBox, boxPlaces];
    }

    function rearrangeBoxesFromMiddle(boxes: Box[]): [number, Box][] {
        /* Change the order of the boxes so that the biggest are in the middle, e.g. 1 3 5 7 8 6 4 2.
        Return the list of tuples (original_index, box). */
        const indexedBoxes: [number, Box][] = boxes.map((box, i) => [i, box]);
        indexedBoxes.sort((a, b) => b[1].weight - a[1].weight);  // sort from heaviest
        let even = [];
        let odd = [];
        for (let i = 0; i < indexedBoxes.length; i++) {
            if (i % 2 == 0) even.push(indexedBoxes[i]);
            else odd.push(indexedBoxes[i]);
        }
        odd.reverse();
        odd.push(...even);
        return odd;
    }

    function connectedComponents(vertices: number[], edges: [number, number][]): number[][] {
        /* Get connected components of a graph (efficient implementation with shallow tree).
        Connected components and the vertices within them are sorted. */
        const shallowTree = newShallowTree(vertices);
        for (const [u, v] of edges) stJoin(shallowTree, u, v);
        return stGetSets(shallowTree);
    }

    function dagConnectedComponents(dag: Dag): Dag[] {
        const components = connectedComponents(dag.vertices, dag.edges);
        const nComponents = components.length;
        const vertex2component = new Map<number, number>();
        for (let iComponent = 0; iComponent < components.length; iComponent++) {
            const component = components[iComponent];
            for (const v of component) vertex2component.set(v, iComponent);
        }
        const dags = [];
        for (let i = 0; i < nComponents; i++) dags.push(newDagEmpty());
        for (let iLevel = 0; iLevel < dag.levels.length; iLevel++) {
            for (const vertex of dag.levels[iLevel]) {
                const iComp = vertex2component.get(vertex)!;
                const levs = dags[iComp].levels;
                if (levs.length <= iLevel) levs.push([]);
                assert(levs.length == iLevel + 1, 'levs.length != iLevel + 1');  // Debug
                levs[iLevel].push(vertex);
                dags[iComp].vertices.push(vertex);
                dags[iComp].in_neighbors.set(vertex, []);
            }
        }
        for (const edge of dag.edges) {
            const [u, v] = edge;
            const iComp = vertex2component.get(u)!;
            assert(vertex2component.get(v) == iComp, 'vertex2component.get(v) != iComp');  // Debug
            dags[iComp].edges.push(edge);
            dags[iComp].in_neighbors.get(v)!.push(u);
        }
        return dags;
    }


    type ShallowTree = Map<number, number>;
    /* Efficient representation of a set of disjoint sets. 
    Each set is identified by its 'root', which is its smallest element.
    The Map maps each element to its root.*/

    function newShallowTree(elements: number[]): ShallowTree {
        let tree = new Map();
        for (const elem of elements) tree.set(elem, elem);
        return tree;
    }
    function stRoot(self: ShallowTree, element: number): number {
        const parent = self.get(element)!;
        if (parent == element) {  // This element is the root
            return element;
        } else {
            const root = stRoot(self, parent);
            self.set(element, root);
            return root;
        }
    }
    function stJoin(self: ShallowTree, i: number, j: number): number {
        /* Join the set containing i and the set containing j, return the root of the new set. (Do nothing if i and j already are in the same set.) */
        const rootI = stRoot(self, i)
        const rootJ = stRoot(self, j)
        const newRoot = Math.min(rootI, rootJ);
        self.set(rootI, newRoot);
        self.set(rootJ, newRoot);
        return newRoot;
    }
    function stGetSets(self: ShallowTree): number[][] {
        /* Return the list of sets, each set itself is a list of elements. */
        const setMap = new Map<number, number[]>();
        for (const elem of self.keys()) {
            const root = stRoot(self, elem);
            if (setMap.has(root)) setMap.get(root)!.push(elem);
            else setMap.set(root, [elem]);
        }
        const setList = [...setMap.values()];
        for (const set of setList) set.sort();
        setList.sort();
        return setList;
    }

    function assert(assertion: boolean, ...details: any): void {
        if (!assertion) {
            console.error('AssertionError:', ...details);
            throw new Error('AssertionError');
        }
    }
}