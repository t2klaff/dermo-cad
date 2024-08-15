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

to run the program using command line:
> java -Djava.library.path=".\opencv\build\java\x64" -jar FYP.jar


- the illegal reflective access error is okay and the program should run just fine

- inside the data folder are the Training, Test, and Output folders, as well as the ground truth metadata and some example arff files
- a few sample images have been included in the Training and Test folders
- any images in the Test folder will have their working-out images written to the Output folder
- the classification results also are written to the output folder
- the training.arff file already includes the data for all 10015 of the HAM10000 instances, this is the file used for the 10-fold cross validation

thanks!