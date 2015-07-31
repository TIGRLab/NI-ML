#!/bin/sh
if [ -z "$1" ];
then
   echo "Usage: ./run_spearmint.sh <path to model>";
   exit 1;
fi

export XP_PATH=$1
export SPEARMINT_DB_ADDRESS=nissl
export SPEARMINT_ROOT=/projects/francisco/repositories/Spearmint/spearmint

#qsub -V -cwd ./_spearmint.sh
./_spearmint.sh
