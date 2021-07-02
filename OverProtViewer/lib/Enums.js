export var Enums;
(function (Enums) {
    let ColorMethod;
    (function (ColorMethod) {
        ColorMethod[ColorMethod["Uniform"] = 0] = "Uniform";
        ColorMethod[ColorMethod["Type"] = 1] = "Type";
        ColorMethod[ColorMethod["Sheet"] = 2] = "Sheet";
        ColorMethod[ColorMethod["Stdev"] = 3] = "Stdev";
    })(ColorMethod = Enums.ColorMethod || (Enums.ColorMethod = {}));
    ;
    let ShapeMethod;
    (function (ShapeMethod) {
        ShapeMethod[ShapeMethod["Rectangle"] = 0] = "Rectangle";
        ShapeMethod[ShapeMethod["SymCdf"] = 1] = "SymCdf";
    })(ShapeMethod = Enums.ShapeMethod || (Enums.ShapeMethod = {}));
    ;
    let LayoutMethod;
    (function (LayoutMethod) {
        LayoutMethod[LayoutMethod["Old"] = 0] = "Old";
        LayoutMethod[LayoutMethod["New"] = 1] = "New";
    })(LayoutMethod = Enums.LayoutMethod || (Enums.LayoutMethod = {}));
    ;
})(Enums || (Enums = {}));
//# sourceMappingURL=Enums.js.map