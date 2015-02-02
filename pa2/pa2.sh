#!/bin/sh

CLASSPATH='bin:lib/*'

# Clean up the directory
mkdir -p bin
find bin -name '*.class' -print0 | xargs -0 rm -f

# Compile the program
find src -name '*.java' -and -not -name '.*' -print0 | xargs -0 javac -cp $CLASSPATH
mv src/*.class bin

# Run the program
echo 'Checking correctness...'
java -javaagent:lib/deuceAgent-1.3.0.jar -cp $CLASSPATH Driver --test=correctness --key-range=1000000 --read-percent=5
java -javaagent:lib/deuceAgent-1.3.0.jar -cp $CLASSPATH Driver --test=correctness --key-range=10000 --read-percent=95
java -javaagent:lib/deuceAgent-1.3.0.jar -cp $CLASSPATH Driver --test=correctness --key-range=1000 --read-percent=99

echo
echo 'Checking performance...'
java -javaagent:lib/deuceAgent-1.3.0.jar -cp $CLASSPATH Driver --test=performance --key-range=1000000 --read-percent=5
java -javaagent:lib/deuceAgent-1.3.0.jar -cp $CLASSPATH Driver --test=performance --key-range=10000 --read-percent=95
java -javaagent:lib/deuceAgent-1.3.0.jar -cp $CLASSPATH Driver --test=performance --key-range=1000 --read-percent=99
