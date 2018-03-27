
# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

### Data Set Summary & Exploration

#### 1. a basic summary of the data set. 

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
[//]: # (Image References)

[image10]: ./examples/train_visual.png "Train Data Visualization"
[image11]: ./examples/valid_visual.png "Valid Data Visualization"
[image12]: ./examples/test_valid.png "Test Data Visualization"

Here is an exploratory visualization of the data set. It is a histogram chart showing how the data distribute:

![alt text][image10]
![alt text][image11]
![alt text][image12]

### Design and Test a Model Architecture

#### 1. Preprocessed Description . 

I normalized the image data because the this will make the model converge fast and make the model more accurate.
Since the data set type is a unsigned short, I use some I have to change it to signed float, so my normalization is a little diffrent, as code showed below: 
``` python
X_train = (X_train / 128.0) - 1.0
X_valid = (X_valid /128.0 ) - 1.0
X_test = (X_test / 128.0)  - 1.0
```

#### 2. Model Architecture Description .

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x6      	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5x16      	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input 400 Output 120        									|
| RELU					|												|
| Fully connected		| Input 120 Output 84        									|
| RELU					|												|
| Fully connected		| Input 84 Output 43        									|
| RELU					|												|
| Softmax				| Get the prediction label        									|
|						|												|
|						|												|
 




#### 3. Trainning Decription

To train the model, I used an Adam Optimizer, the batch size is 128 ,the number of epochs is 100 and Learning rate is 0.001, and the drop out is 50% while trainning in the first fullly connected layer


#### 4. Solution Description

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.963
* test set accuracy of 0.953

Since the original LeNet is not fit the traffic sign images, So I changed the input channels to 43 and the output classes to 43. 
So the LeNet can be used to train and predict the traffic sign images. 
But the tranning converge process is slow and the accuracy of test data is about 0.89. So I normalize the data and the accuracy is about 0.93 or lower, But the accuracy of training data is about 1.0 and the validation data accuracy is 0.96, this indicate the model is overfitted, So a 0.5 drop out is introducted in the first full connected layer
Finally the raining set accuracy of 1.0, validation set accuracy of 0.963,test set accuracy of 0.953.

### Test a Model on New Images

#### 1. New Data

Here are 8 German traffic signs that I found on the web:

[//]: # (Image References)

[image20]: ./test_images/4 "70"
1st pitcture missing some pixels, it is even hard for human to classify

[image21]: ./test_images/7 "100"
2nd piture is easy to mixed up with 80km/h

[image22]: ./test_images/14 "Stop"

3rd image's lightness is low 

[image23]: ./test_images/17 "No Entry"

4th image is blur

[image24]: ./test_images/22 "Bumpy"

5th image's lightness is low

[image25]: ./test_images/23 "Slippy"

6th image's lightness is high


[image26]: ./test_images/35 "Straight"

7th image's blur

[image27]: ./test_images/40 "Roudabout"
8th image's backgroud is a little complicated



![alt text][image20] ![alt text][image21] ![alt text][image22] ![alt text][image23] 
![alt text][image24] ![alt text][image25] ![alt text][image26] ![alt text][image27] 

#### 2. Model's Predictions on New Images
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 14 Stop Sign      		| 14 Stop sign   									| 
| 23 Slippery Road     		| 23 Slippery Road 										|
| 17 No Entry				| 17 No Entry											|
| 4  70 km/h	         	| 4  70 km/h					 				|
| 7  100 km/h		        | 7  100km/h     							|
| 40 Roundabout     		| 40 Roundabout   									| 
| 35 Ahead Only     		| 35 Ahead Only 										|
| 22 Bumpy Road				| 22 Bumpy Road											|


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100.0%. This compares favorably to the accuracy on the test set of **95.3%**

#### 3. Certainty Description

For the 1st image, the model is very sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|

For the 2rd image, the model is very sure that this is a Slippery Road sign (probability of 1.0),,The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Slippery Road   								| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|

For the 3rd image, the model is very sure that this is a No Entry sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No Entry  									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|

For the 4th image, the model is very sure that this is a  Speed Limit 70 km/h sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 70 km/h   									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|

For the 5th image, the model is very sure that this is a Speed Limit  sign 100 km/h (probability of 1.0)，The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| 100km/h   									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|

For the 6th image, the model is very sure that this is a Roundabout sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Roundabout  									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|

For the 7th image, the model is very sure that this is a Ahead Only sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Ahead Only									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|

For the 8th image, the model is very sure that this is a Bumpy Road sign (probability of 1.0), The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Bumpy Road  									| 
| 0.0     				| 20km/h 										|
| 0.0					| 30km/h										|
| 0.0	      			| 50km/h		     			 				|
| 0.0				    | 60kmk/h      	         						|
