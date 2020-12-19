# Implementation of the Bag of Words Model.

## Bag of Words Model

The purpose of this assignment is to create a bag of words model. The training data consisted of data in the form of images of 4 different types of objects. That is images of an accordion, a dollar bill, a motorbike and a soccer ball. Each type consists of several images of its kind. 

In the bag of words model, image features are taken into consideration, and similar features are grouped together. Thus, in order to implement this bag of words model, we need to extract features of each image. Also, we can extract the description of every image. 

This extraction of the features and the description of every image is achieved using the SIFT code. SIFT is a Scale Invariant Feature Transformation method, that identifies the local features in an image along with its description. To achieve this, internally SIFT performs – Scale Space Extrema Detection, Key Point Localization, Orientation Assignment, Key  Point Descriptor and, Key Point Matching.

## Working 

•	In this assignment, I have read the images using cv2.imread(image) function. I further convert this image to grayscale using cv2.cvtColor(image, cv2.COLOR_BR2RAY) function. 
•	Further, I used the sift code as seen in figure 1 to detect the features. The method used is  obj_sift.detectAndCompute(). 
•	The cv2.drawKeypoints() function is used to draw the identified key points on the original image.

![Fig1](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig1.png?raw=true)

•	K- means clustering - An important part of this assignment is the use of K-means Clustering algorithm to categorize data into different clusters. (I have used 100 clusters for this assignment). The purpose is to form groups of features with similar properties.
•	The function used is km_obj.fit_predict(). The input passed to this function is a vertical stack consisting of the descriptions for every feature of every image.
•	Further, I generate histogram based on the built bag of words. 

## Results of images

The following images help visualize the features considered by SIFT. One result image for every type of image is shown below.

![Fig2](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig2.png?raw=true)

![Fig3](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig3.png?raw=true)

![Fig4](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig4.png?raw=true)

![Fig5](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig5.png?raw=true)

## Results of  Visualization using Histogram

**Type 1 – Histogram (Accordion)**

![Fig6](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig6.png?raw=true)

**Type 2 – Histogram (Dollar_bill)**

![Fig7](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig7.png?raw=true)

**Type 3 – Histogram (Motorbike)**

![Fig8](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig8.png?raw=true)

**Type 4 – Histogram (Soccer_Ball)**

![Fig9](https://github.com/AnkitaKhadsare95/Computer-Vision/blob/main/Bag%20Of%20Words%20Model/Images/Fig9.png?raw=true)
