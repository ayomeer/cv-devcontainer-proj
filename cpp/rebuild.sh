#!/bin/bash

tmp_cwd=$(pwd)

cd /app/cpp/build
rm -rf ./*
cmake ..
make
cp cppmodule.cpython-310-x86_64-linux-gnu.so /app/python/lib
echo "Rebuilt and copied cppmodule to /app/python/lib"

cd $tmp_cwd
