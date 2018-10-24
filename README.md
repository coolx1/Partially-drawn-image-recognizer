# Partially drawn image recognizer
This project is an extension to the WebUI image classifier. https://github.com/coolx1/WebUI-element-classifier.
Recognizes partially drawn images of the 11 classes of images and tells the closest match among the 11 classes of images. 
This project uses the output from the fully connected layer of the CNN used in the WebUI image classifier project.
For every partially drawn image the output from the fully connected layer is seen as a vector representation of the image. This vector is
compared with a set of standard images from each class using vector similarity measures like Cosine siimilarity and Euclidean distance and
outputs the closest match. 

Run the partialImageClassifier.py by changing the filename variable in the python file to any image in the sample images to test the model's predictions.
Uncomment the Euclidean distance as similarity measure portion and comment out the Cosine similarity as similarity measure portion to find out predictions
using the other similarity measure.

This project contains the following files:
## Segmentation.py 
   Contains the code used for segmentation of images. This is the same as the one used in
   the UI Image classifier.
## sample_diagrams
   contains sample images to test the partialImageClassifier.
## baseDiagrams.npy
   contains the base diagrams from every class with which the partially drawn images are
   compared(Euclidean and cosine similarities are taken.)
## model1.json
   contains the model of the same cnn as that of the UI Image classifier except that the output
   layer is removed.
## model1.h5 
   contains the weights for model1.json
## partialImageClassifier.py
   contains the code to test the partial Image classifier.
