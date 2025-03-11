#!/bin/bash

JOB_NAME="${1}_${2}_${3}"

OUTFILE="${JOB_NAME}.%j.out"
ERRFILE="${JOB_NAME}.%j.err"

sbatch -J "${JOB_NAME}" -o "${OUTFILE}" -e "${ERRFILE}" scripts/train.sh $1 $2 $3 -steps 10000000