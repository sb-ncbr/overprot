# OverProtViewer

**OverProtViewer** is a tool for visualization of secondary structure consensus for protein families (produced by **OverProt**) in a web browser.

## Build

    sh build.sh

Built files will be in `dist/` directory.

## Run locally

    node local_server.js

Then open http://localhost:3000 in web browser

## Include in a web

Minimal example:

```html
<html>
<head>
    <meta charset='UTF-8'>
    <link rel='stylesheet' type='text/css' href='dist/overprot-viewer.min.css'>
    <script type='text/javascript' src='dist/d3.v5.min.js'></script>
    <script type='text/javascript' src='dist/d3-selection-multi.v1.min.js'></script>
    <script type='text/javascript' src='dist/overprot-viewer.min.js'></script>
</head>
<body>
    <overprot-viewer id="rossmann"
        file="sample_data/rossmann_3-40-50-2300/diagram.json"
        width="1200" height="300"
        color-method="sheet" shape-method="rectangle" 
        beta-connectivity="true" occurrence-threshold="0.2"
        dispatch-events="false" listen-events="false">
    </overprot-viewer>
</body>
</html>
```

Allowed values for attributes:

- **`file`**: URL of the JSON file with preprocessed input data (mandatory)
- **`width`**, **`height`**: positive integers
- **`color-method`**: `uniform`, `type`, `sheet`, `variability`, `rainbow`
- **`shape-method`**: `rectangle`, `symcdf`
- **`beta-connectivity`**: `true`, `false`
- **`occurrence-threshold`**: float between 0.0 and 1.0
- **`dispatch-events`**: `true`, `false`
- **`listen-events`**: `true`, `false`

The above example sets all optional attributes to their default values (could be omitted).

See `web/example-events.html` for interconnecting OverProt Viewer with other elements in a web page.

## Deployed instance

https://overprot.ncbr.muni.cz
