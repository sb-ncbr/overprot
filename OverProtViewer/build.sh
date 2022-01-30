#!/bin/sh
set -e

rm -rf lib/
rm -rf dist/

tsc

./node_modules/.bin/rollup -c

cp src/overprot-viewer.css dist/
cp libraries/* dist/

./node_modules/node-minify/bin/cli.js  -c uglify-es  -i dist/overprot-viewer.js  -o dist/overprot-viewer.min.js
./node_modules/node-minify/bin/cli.js  -c html-minifier  -i dist/overprot-viewer.css  -o dist/overprot-viewer.min.css

echo BUILD FINISHED
