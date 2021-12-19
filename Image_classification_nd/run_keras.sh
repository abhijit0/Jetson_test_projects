#!/bin/bash

python3 run_test_keras.py &> /dev/null

input="logs_keras.log"
while IFS= read -r line
do
  echo "$line"
done < "$input"