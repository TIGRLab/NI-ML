#!/bin/sh
if [ -z "$1" ];
then
   echo "Usage: ./clean_spearmint.sh <path to model>";
   exit 1;
fi
CWD=$(pwd)

export XP_PATH=$CWD/$1
export SPEARMINT_DB_ADDRESS=nissl
export SPEARMINT_ROOT=/projects/francisco/repositories/Spearmint/spearmint
cd $SPEARMINT_ROOT
if [ -d $XP_PATH ]
then
  python -c "import spearmint.utils.cleanup as cleanup; import sys; cleanup.cleanup(sys.argv[1])" $XP_PATH
  cd $XP_PATH
  echo "Removing old logs from $XP_PATH/output/"
  rm ./output/*
  cd $CWD
else
  echo "$XP_PATH Not a valid directory."
fi
