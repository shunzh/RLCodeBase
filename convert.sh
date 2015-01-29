#!/bin/bash
for f in `ls task*.eps`; do
  convert $f -density 100 -flatten ${f%.*}.png;
done
