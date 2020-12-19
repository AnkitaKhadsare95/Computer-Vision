__author__='AAK'

'''
CSCI-631: Foundations of Computer Vision
Author: Ankita Anilkumar Khadsare [ak8932] 

Homework 3: Bag of words model.
'''

import cv2
import os
import numpy as np
from matplotlib import pyplot as pt
from sklearn.cluster import KMeans


class BagOfWords:

    __slots__ = 'type_dict', 'type_count', 'features_dict', 'feature_descriptions',\
                'stack', 'image_count', 'km_output', 'histogram', 'clusters','images'


    def __init__(self):
        '''
        Initialization
        '''

        self.type_count = 0;
        self.image_count = 0;
        self.features_dict = dict()
        self.feature_descriptions = []
        self.stack = None
        self.clusters = 100
        self.km_output = None
        self.histogram = None
        self.type_dict = dict()
        self.images=[]


    def plot_Histogram(self):
        '''
        Plotting Histogram for individual image
        :return: None
        '''

        self.histogram = np.array([np.zeros(self.clusters) for i in range(self.image_count)])
        old_count = 0
        for iter1 in range(self.image_count):
            size = len(self.feature_descriptions[iter1])
            for iter2 in range(size):
                num = self.km_output[old_count+iter2]
                self.histogram[iter1][num] +=1
            old_count += size

        i=1
        for iter1 in range(len(self.histogram)):
            if iter1%14 == 0:
                pt.bar([k for k in range(len(self.histogram[iter1]))], self.histogram[iter1])
                pt.xlabel("Clusters")
                pt.ylabel("Number of features")
                pt.title(str(self.images[iter1]))
                pt.savefig("Histogram of Image"+str(i))
                i +=1
                pt.show()


    def train(self, train_data):
        '''
        Feature extraction using SIFT
        :param train_data: raw data
        :return: None
        '''

        new_dir = 'OutputImages'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        for type in os.listdir(train_data):
            self.type_dict[str(self.type_count)]=type
            for img_name in os.listdir(train_data+'/'+type):
                imgn = img_name
                img_name = train_data+'/'+type+'/'+img_name
                # print(img_name)
                self.images.append(img_name)
                img_color = cv2.imread(img_name)
                img_grayscale = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                # extracting images using sift
                obj_sift = cv2.xfeatures2d.SIFT_create()
                features, feature_desc = obj_sift.detectAndCompute(img_grayscale, None)
                img_features = cv2.drawKeypoints(img_grayscale, features, cv2.imread(img_name).copy())

                new_img_path= new_dir+'/'+type+'_'+ imgn   #os.path.join(new_dir, type+'_'+img_name)
                print('Obtained Output Image : ',new_img_path)
                # writing image to new file
                cv2.imwrite(new_img_path, img_features)

                self.feature_descriptions.append(feature_desc)
                self.image_count +=1;
                if self.features_dict.__contains__(type):
                    self.features_dict[type].append([img_name, cv2.imread(img_name), features, feature_desc])
                else:
                    self.features_dict[type] = [[img_name, cv2.imread(img_name), features, feature_desc]]
            self.type_count +=1

        # converting features_description list to stack
        self.stack = np.array(self.feature_descriptions[0])
        for remaining in self.feature_descriptions[1:]:
            self.stack = np.vstack((self.stack, remaining))

        # print("Size of stack", len((self.stack)))

        # applying kmeans and building the vocabulary
        km = KMeans(n_clusters=self.clusters)
        self.km_output = km.fit_predict(self.stack).copy();


def main():
    '''
    The main function
    :return:  None
    '''
    obj = BagOfWords()
    train_data = input('Enter training data path i.e., path of the extracted train folder:')
    obj.train(train_data)
    obj.plot_Histogram()

if __name__ == '__main__':
    main()