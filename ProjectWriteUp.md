# **Project - Traffic Sign Recognition WriteUp** 

---

### Overview

In this project, I leverage convolutional neural network to build a traffic sign classifier which can be used to recoginize traffic signs for self-driving/autonomous cars.

The goals / steps of this project are as follows:
* Load the data set [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./ImageFile/image1.png "LeNet5-Accuracy"
[image2]: ./ImageFile/image2.png "LeNet5-Loss"
[image3]: ./ImageFile/img3.png "Bar-Train"
[image4]: ./ImageFile/img4.png "Bar-Valid"
[image5]: ./ImageFile/img5.png "Bar-Test"
[image6]: ./ImageFile/img6.png "Original"
[image7]: ./ImageFile/img7.png "Normalized"
[image8]: ./ImageFile/img8.png "Result"
[1]: ./ImageFile/1.jpg "Test-1"
[2]: ./ImageFile/2.jpg "Test-2"
[3]: ./ImageFile/3.jpg "Test-3"
[4]: ./ImageFile/4.jpg "Test-4"
[5]: ./ImageFile/5.jpg "Test-5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 

  34799
  
* The size of the validation set is ?

  4410

* The size of test set is ?

  12630

* The shape of a traffic sign image is ?

  (32, 32, 3)

* The number of unique classes/labels in the data set is ?

  43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.

The bar charts below show the frequecy of each traffic sign class among the training, validation and test sets.

![3][image3]

![4][image4]

![5][image5]

From the charts above, we can see that the traffic sign class distributes similarly among the three datasets.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data by dividing each pixel by its maxmimum value. For example, for images in the training set for each pixel, I divide it by the maximum value among training set.

This normalization makes each pixel has value between 0-1. The reason to apply this normalization is because

  * Original value between 0 and 255 may cause several directions with large pixiel values dominate others in gradient descent
  * This normalization doesn't change the visualization of the traffic sign for human eyes

Here is an example of the original traffic sign and the normalized ones:

Original

![Original][image6]

Normalized

![Normalized][image7]

 
In addition, I also tried convert the color image to grayscale image but this doesn't improve model's performance.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride,  outputs 10x10x16   		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flattened		| 14x14x6 + 5x5x16 (low level convolution + high level convolution)|
| Fully Connected |             120                |
| Fully Connected |             84             |
| Softmax				| 43								|

In addition to the network architecture above, please note the following as well

  * The (data) loss function is cross-entropy because this is a classification problem.
  * L1 loss is added to the data loss. The final loss function is data loss plus L1 loss
  

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The parameters related to training is as follows:

  * Optimizer: Adam
  * Batch size: 256
  * Number of Epochs : 150
  * Learning Rate: 0.01
  * Regularization Coefficient(L1): 0.000205

Most optimizer, batch size and learning rate above are not well tuned. The default value is from my previous experience training neural network. The number of epochs is obtained in a heuristic way by checking training-validation accuracy graph. The L1 coefficient is obtained through validation.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 0.98
* validation set accuracy of ? 0.94
* test set accuracy of ? 0.93

If an iterative approach was chosen:
* **What was the first architecture that was tried and why was it chosen?**
  
  I started from the LeNet-5 because 1) it is reported that LeNet-5 can achieve 0.89 accuracy on validation set which in my opinion a good starting point 2) based on my experiment, LeNet-5 can achieve 0.99 accuracy on training set which demonstrates that LeNet-5 is a low-bias model. By considering 1) and 2), it is possible to improve LeNet-5's performance on validation set (to the required 0.93) by adding regularity into the architecture (L1/L2 norm or dropout).

* **What were some problems with the initial architecture?**

  As mentioned above, based on my experiment (see plot below), LeNet-5 suffers from high variance. We have to reduce the variance. 

  ![][image1]
  
  ![][image2]


* **How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**

  The initial architecture is adjusted in the following two important aspects:
    * The first fully-connected layer in LeNet-5 is only connected to the second convolution layer (high level features). In my model,         The first fully-connected layer is also connected to the first convolution layer (low level features). This architecture is used         because LeNet-5 isn't powerful enough to obtain a 0.93 or higher validation accuracy and it is reported by Yann Lecun that      combining low and high level feature leads to a more powerful network.
    
    * L1 regularization term is introduced in the network to decrease the variance hence increase validation accuracy

* **Which parameters were tuned? How were they adjusted and why?**

  L1 coefficient is tuned based on bias-variance tradeoff shown in the graph
  
  ![][image8]
   
* **What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**

  Same as question above, the important design choices are
  
  * The first fully-connected layer in LeNet-5 is only connected to the second convolution layer (high level features). In my model,         The first fully-connected layer is also connected to the first convolution layer (low level features). This architecture is used         because LeNet-5 isn't powerful enough to obtain a 0.93 or higher validation accuracy and it is reported by Yann Lecun that      combining low and high level feature leads to a more powerful network.
    
   * L1 regularization term is introduced in the network to decrease the variance hence increase validation accuracy



 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][1]

* Difficulty
  * The stop sign isn't facing directly toward the camera (tangential distortion)
  

![alt text][2]

* Difficulty
  * Tangential distortion
  * Watermark on the image

![alt text][3] 

* Difficulty
  * The stop sign only takes small portion of the whole image


![alt text][4]
* Difficulty
  * Watermark
  * The background buildings look like connected to the sign


![alt text][5]

* Difficulty
  * Too much background objects



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Speed Limit (60km)   									| 
| Road Work     			| Slippery Road 										|
| Speed Limit (50km)					| End of all speed and passing limits											|
| Road Work	      		| Slippery Road					 				|
| No Entry			| Slippery Road      							|


The model was able to correctly guess 0 of the 5 traffic signs, which gives an accuracy of 0%. This this much worse than its performance on testing set. I think the reason is bacause these new images are from difference sources than the German Traffic Sign dataset, which makes recognition much more difficult.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 18th-23rd cell of the Ipython notebook.

For the first image, the model is relatively sure the sign among Speed Limit (60km), Bumpy Road or Stop sign. However, the true sign is stop sign. The model makes wrong prediction but the top three includes the correct sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .22         			| Speed Limit(60km)   									 
| .21     				| Bumpy Road										
| .16					| Stop											
| .10	      			| 	Dangerous curve to the right  
| .08				    | Yield      							|


For the second image, the model is 100% sure the sign is slippery sign. However, the true sign is road work

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Slippery road |

For the third image, the model is pretty sure the sign is End of all speed and passing limits . However, the true sign is speed limit (50km). The top five prediction include speed limit sign but with wrong speed value.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| End of all speed and passing limits  
| .01     				| Speed limit (30km/h) 
| .01					| End of no passing											
| .0	      			| 	End of speed limit (80km/h)
| .0				    | Roundabout mandatory     							|

For the fourth image, the model is pretty sure the sign is slippery road . However, the true sign is road work. The second prediction is road work. Based on the result on the second and the fourth image, we can see the model tends to mis-classify road work sign as slippery sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .74         			| Slippery road  
| .25     				| Road work
| .0					| Right-of-way at the next intersection								
| .0	      			| 	Wild animals crossing
| .0				    |     Dangerous curve to the right 							|

For the fifth image, the model is pretty sure the sign is slippery road . However, the true sign is no passing. The top five prediction doesn't include the correct sign.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97         			| Slippery road    
| .02     				| Road work
| .01					| Right-of-way at the next intersection											
| .0	      			| 	Dangerous curve to the left
| .0				    | Keep right     							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


