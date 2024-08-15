# dermo-cad

## Abstract

The skin is the body’s largest organ accounting for roughly 16% of total body weight, and 
just like any body part, the cells that comprise the skin can develop cancers. Skin cancer is 
one of the most common cancers, and some variants are life-threatening. Most people have 
between 20 and 50 pigmented skin lesions, and although most are harmless, roughly 1 in 
33,000 skin lesions become cancerous. Early diagnosis is key for cancer survival, and 
computer-aided diagnosis (CAD) systems intend to save lives by helping speed up the 
process of obtaining a diagnosis. This report details the design and implementation of a CAD 
system for pigmented skin lesions. The proposed system can classify dermoscopic images of 
skin cancer with upwards of 83.3% accuracy using a Random Forest classifier. The best 
results were achieved with 10-fold cross validation on the HAM10000 dataset; the reported 
accuracy was 89.56%, the reported F1 score was 0.895, the MCC score was 0.794, and the 
ROC Area score was 0.962. These results were achieved using the full feature set, including 
88 features describing each lesion in terms of its asymmetry, border structure, colour, and 
differential structures, according to the ABCD dermoscopy algorithm.

Full report with references included __(FYP - Final Report.pdf)__

### Prerequisites:

Install OpenCV version 4.5.5 in the same location as FYP.jar (newer versions _might_ work)

Weka and SMOTE dependencies are included in FYP.jar




### The directory should look like:
```
root...
 \bin\
 \data\
 \opencv\build\java...    <--
 \src\
 \FYP.jar
 \help.jpg
 \readme.txt
```

### Run the program using:
> java -Djava.library.path=".\opencv\build\java\x64" -jar FYP.jar
