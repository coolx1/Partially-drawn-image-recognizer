Project - Partially drawn image Identification - project 2

This project contains the following files:
1) Segmentation.py contains the code used for segmentation of images. This is the same as the one used in
   the UI Image classifier.
2) sample1, sample2,.....sample10.png  are sample images to test the partialImageClassifier.
3) baseDiagrams.npy contains the base diagrams from every class with which the partially drawn images are
   compared(Euclidean and cosine similarities are taken.)
4) model1.json contains the model of the same cnn as that of the UI Image classifier except that the output
   layer is removed.
5) model1.h5 contains the weights for model1.json
6) partialImageClassifier.py - contains the code to test the partial Image classifier.