#!/bin/sh

# Dependencies:
# sudo apt install node
# sudo apt install node-typescript
# sudo apt install npm ?
# npm install ?

rm -rf lib/
rm -rf dist/

tsc  &&

./node_modules/.bin/rollup -c  &&

cp src/overprot-viewer.css dist/  &&
cp libraries/* dist/  &&
# cp web/* dist/

./node_modules/node-minify/bin/cli.js  -c uglify-es  -i dist/overprot-viewer.js  -o dist/overprot-viewer.min.js  &&
./node_modules/node-minify/bin/cli.js  -c html-minifier  -i dist/overprot-viewer.css  -o dist/overprot-viewer.min.css  &&

echo BUILD FINISHED