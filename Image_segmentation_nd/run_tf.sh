#!/bin/bash
 
while getopts ":d:b:" opt; do
  case $opt in
    d) debug=$OPTARG
      ;;
    b) batch_size=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ $debug == "yes" ]]
then
  input="logs_tf.log"
  if [[ -f $input ]]
  then 
    rm -rf $input
  fi 
  python3 Image_segmentation_tf.py $batch_size
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
  
elif [[ $debug == "no" ]]
then
  input="logs_tf.log"
  if [[ -f $input ]]
  then 
    rm -rf $input
  fi
  python3 Image_segmentation_tf.py $batch_size &> /dev/null
  while IFS= read -r line
  do
    echo "$line"
  done < "$input"
else
  echo "Please provide no-debug or debug option"
fi