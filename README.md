# dermo-cad

to run the program:
	- install OpenCV version 4.5.5 in the same location as FYP.jar
	- Weka and SMOTE dependencies are included in FYP.jar
	- the directory should look like:
-----
root...
   \bin\
   \data\
   \opencv\build\java...    <--
   \src\
   \FYP.jar
   \help.jpg
   \readme.txt
-----

run the program using command line:
> java -Djava.library.path=".\opencv\build\java\x64" -jar FYP.jar