This is a project I started in June 2019 (coming out of my first year in university) and completed in July to introduce myself to deep 
learning and how neural networks work.
I learned all I needed to start this project from 3Blue1Brown's playlist on the topic (https://www.3blue1brown.com/neural-networks), 
and by reading through Michael Nielsen's book on *Neural Networks and Deep Learning* (http://neuralnetworksanddeeplearning.com/).

### Summary:
This is a Convolutional Neural Network I built to detect numbers from a webcam. It is all programmed in Java from scratch using only
a linear algebra library to help with matrix multiplication and OpenCV to take in pixel values from my webcam. 
From knowing nothing about artificial intelligence, I taught myself how to do this in a month through online research and taking 
extensive notes. I even built a fully functional GUI for it that displayed the camera view and the number it read in real-time.

### Where to find relevant code:
**NETWORK:**
CNN_mnist/src/main/java/ConnectedNetwork/
  - Network: This is the network class where a CNN is stored for training and testing.
  - networkMain: This is the main class for operations to be done from the networkOperations file.
  - networkOperations: This file contains all the methods that can be performed on a network in a clean manner so that it can be 
        called in the networkMain file (methods such as training, testing, getting serialized network, etc..).

**DATA:** 
CNN_mnist/src/main/java/MNISTReading/
  - MnistDataReader: class to read and parse through the Mnist data to provide meaningful inputs and labels for our network.
  - MnistMatrix: class that holds the actual data for each image.

**GUI:**
CNN_mnist/src/main/java/GUI/
 - Display: This is the file to run if you want to see the network work in real-time. I have already included some serialized networks 
      that have been trained to be used for this purpose.
