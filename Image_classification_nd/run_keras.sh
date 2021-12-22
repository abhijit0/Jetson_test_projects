#!/bin/bash

option=$1
if [[ $option == "-debug" ]]
then 
  input="logs_keras.log"
  rm -rf $input
  python3 run_test_keras.py 
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
elif [[ $option == "-no-debug" ]]
then
  input="logs_keras.log"
  rm -rf $input
  python3 run_test_keras.py &> /dev/null
  input="logs_keras.log"
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
else
  echo "Please provide no-debug or debug option"
fi