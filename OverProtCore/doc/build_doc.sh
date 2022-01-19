#!/bin/bash
set -e
DIR=$(dirname $0)
cd $DIR

NAME="pica"

xelatex $NAME.tex  \
&&  xelatex $NAME.tex  \
&&  rm $NAME.aux $NAME.log $NAME.out  $NAME.toc  \
&&  echo "OK  $NAME.pdf"
