#!/bin/bash

option=$1
if [[ $option == "-debug" ]]
then
  input="optimized/logs_trt.log"
  rm -rf $input 
  python3 optimized/run_test.py
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
  
elif [[ $option == "-no-debug" ]]
then
  input="optimized/logs_trt.log"
  rm -rf $input
  python3 optimized/run_test.py &> /dev/null
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
else
  echo "Please provide no-debug or debug option"
fi