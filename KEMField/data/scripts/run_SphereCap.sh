#!/bin/sh

for i in {1..25}
do
    bin/TestSphereCap_GMSH ${runtype} 2
done
