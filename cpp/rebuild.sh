#!/bin/bash

cd ./build
rm -rf ./*
cmake ..
make
cp cppmodule.cpython-310-x86_64-linux-gnu.so /app/python/lib
echo replaced cppmodule in /app/python/lib
