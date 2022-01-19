#!/bin/bash
set -e
SCRIPT_DIR=$(dirname $0)
cd $SCRIPT_DIR

INFILE="../Description_of_methods.md"
# OUTFILE="../Description_of_methods.pdf"
OUTFILE="../Description_of_methods.tex"

AUX_INFILE="$(dirname $INFILE)/tmp-$(basename $INFILE)"

cat $INFILE | sed -E '/^# *\b.*$/d' | sed -E 's/^##/#/' > $AUX_INFILE

pandoc $AUX_INFILE \
    -o $OUTFILE \
    -H header.tex \
    --include-before-body cover.tex \
    -V geometry:margin=1in \
    --toc \
    --toc-depth 3 \
    -V toc-title:"Table of contents" \
    --number-sections \
    -V classoption:draft \
    --pdf-engine xelatex \
    -V mainfont="DejaVu Serif" \
    -V monofont="DejaVu Sans Mono" \
    -V fontsize=12pt \
    # -V mathfont="DejaVu Math TeX Gyre" \
    # -V documentclass:article \
    # --top-level-division=section \
    # --metadata=title:"OverProt - Description of methods kunda" \
    # --metadata=author:"Adam Midlik, Ivana Hutařová Vařeková, Jan Hutař, Aliaksei Chareshneu," \
    # --metadata=author:"Karel Berka, and Radka Svobodová" \
    # --metadata=lang:"en-US" \
    # --metadata=cover-image:"cover.png" \

rm $AUX_INFILE
