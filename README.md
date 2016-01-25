# img_objects_detector

This project is the detector of specific objects on images (e.g. airplanes). Completed as a part of MSU University course of Computer graphics.
Main file of the program is *src/task2.cpp*, which contains the main program that can be executed in two modes: binary classifier and multi-class classifier.

Program performs several steps on the training stage and the prediction stage.
Training stage:
* loads images of the training set (data/binary/train or data/multiclass/train) with their labels (data/binary/train_labels.txt or data/multiclass/train_labels.txt) 
* extracts features
* trains SVM classifier
* prints model in a file

Testing stage:
* loads images of the test set
* prints results based on the prepared model

Script *compare.py* is provided which can measure accuracy of completed classification.

To compile the program, run in the command shell from the project's root directory
```
make all
```

To pass through training and testing stage, go to *build/bin/* and run
```
task2.exe -d ../../data/binary/train_labels.txt -m model.txt --train
task2.exe -d ../../data/binary/test_labels.txt -m model.txt -l predictions.txt --predict
```

To remove compiled executables, run in the command shell from the project's root directory
```
make clean
```
