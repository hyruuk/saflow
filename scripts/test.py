#!/bin/bash
for SUBID in 04 05 06 07 08 09 10 11 12 13 14
do
sbatch submission_connectivity.sh $SUBID
done