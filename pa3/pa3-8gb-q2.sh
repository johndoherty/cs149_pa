#!/bin/sh

HADOOP_HOME="/usr/local/hadoop-1.2.1"

# Clean up the directory
find . -name '*.class' -print0 | xargs -0 rm -f
mkdir -p class_dir

# Compile the program
find . -name '*.java' -and -not -name '.*' -print0 | xargs -0 javac -cp "${HADOOP_HOME}/hadoop-core-1.2.1.jar" -d class_dir

jar -cvf ngram.jar -C class_dir .

hadoop fs -put query1.txt .
hadoop fs -put query2.txt .
hadoop fs -rmr output
hadoop jar ngram.jar Ngram 4 query2.txt /wikipedia/8gb output
rm -rf output
hadoop fs -get output .
cat output/part-*
