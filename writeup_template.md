#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/road_work_25.png "Road Work 25"
[image5]: ./examples/keep_right_38.png "Keep Right 38"
[image6]: ./examples/priority_road_12.png "Priority Road 12"
[image7]: ./examples/dangerous_curve_left_19.png "Dangerous Curve Left 19"
[image8]: ./examples/Speed_limit_60_3-0.png "Speed Limit 60"
[image9]: ./examples/statistics.png "Class Label Histogram"
[image10]: ./examples/processed.png "Brightness Augmentation"
[image11]: ./examples/spatial_transform_augmentation.png "Spatial Augmentation"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/wpliu25/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell (Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas: In [177]) of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code fil
The code for this step is contained in the third code cell (Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas: In [180]) of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing a histogram of the labels in the original training data.

![alt text][image9]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth through 6th code cell of the IPython notebook.

As a first step, I normalized the image data from RGB values of 0 - 255 to 0.0 to 1.0 because this allowed me to use use matplotlib's conversion from RGB to HSV, explained in the second step. 

As a second step, I decided to convert images to HSV to implement brightness augmentation. By adding random brightness to the images the models should learn not to rely on brightness for sign classification.

Here is an example of a traffic sign image before and after processing: normalization + brightness augmentation.

![alt text][image10]

As a last step, I normalized the image data from RGB values of 0.0 to 1.0 to -0.5 to 0.5 after several iterations where these changes improved accuracy. 

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the nineth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set (80%) and validation set(20%). I did this by using the following:
from sklearn.model_selection import train_test_split

training_data,validation_data,training_label,validation_label=train_test_split(X_train_p,y_train,test_size=0.2)


My final training set had X number of images. My validation set and test set had Y and Z number of images.

* The size of the final augmented training set is 40360
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

The fourth through sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data after looking at the histogram of the training dataset which had uneven distribution of labels. To add more data to the the data set, I used random 2D translation of up to -5 to 5 pixels (left and right) as well as random rotation of up to -45 to 45 degrees. These spatial transformations were chosen to match real world scenarios of variation of the image of the sign as captured by the car's camera.

Here are example of 10 original image and it's randomly augmented image for translation and rotation:

![alt text][image11]

The difference between the original data set and the augmented data set is the following ... 
All labels less than 1000 were augmented 30% of their difference between existing count and 1000.
ex: label 1 only had 180 to start and was augmented with 270 more examples, i.e. floor((1000-180)*0.33) = 270

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model, using LeNet, consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, Valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| Input 400 Output = 320        									|
| RELU					|												|
| Fully connected		| Input 320 Output = 120        									|
| RELU					|												|
| Fully connected		| Input 120 Output = 84        									|
| RELU					|												|
| Fully connected		| Input 84 Output = number of classes = 43        									|
|						|												|
|						|												|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used the following:
* 200     epochs
* 128    batch size
* 0.0009 learning rate
* AdamOptimizer, Kingma and Ba's Adam algorithm. Advantages provided by Adam over GradientDescentOptimizer includes uses moving averages of the parameters (momentum). Disadvantages of Adam is that it requires more computation to be performed for each parameter in each training step and more state to be retained for each parameter.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the eighth and nineth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.95 
* test set accuracy of 0.924

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet was first chosen due to familiarity and simplicity. I understood all layers and thought that LeNet would provide a good benchmark. It is a pioneering model that was first applied to classifies digits and worked well 'out-of-the-box' for classifying traffic signs.
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I added an additional fully connected layer in order to decrease the number linear classifiers at a slower rate.
Original: 2 layers were used to decrease fully connection layers from 400->120->84->43
Final:    3 layers were used to decrease fully connection layers from 400->320->120->84->43
* Which parameters were tuned? How were they adjusted and why?
I adjusted the learning rate slightly from 0.001 to 0.0009 which made a small improvement. This might be due to small step size, as imposed by the learning rate to 
* What are some of the important design choices and why were they chosen? 
2 convolutional layers should first learn to recognize edges then shape, respectively. This was hypothesized to work well as significant features of road signs.

If a well known architecture was chosen:
(see above regarding LeNet)


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The image, 'Dangerous Curve Left 19',  might be difficult to classify because of the black background, which does not reflect training backgroudns of typical natural or urban scenes in daylight. Similarly, the 'Keep Right 38' has a difficult background of snow and noise.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:
[image4]: ./examples/road_work_25.png "Road Work 25"
[image5]: ./examples/keep_right_38.png "Keep Right 38"
[image6]: ./examples/priority_road_12.png "Priority Road 12"
[image7]: ./examples/dangerous_curve_left_19.png "Dangerous Curve Left 19"
[image8]: ./examples/Speed_limit_60_3-0.png "Speed Limit 60"

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work 25      		| Road Work 25  									| 
| Keep Right 38    			| Speed Limit 60 										|
| Priority Road 12					| Priority Road 12										|
| Dangerous Curve Left 19      		| 	Road Work 25			 				|
| Speed Limit 60		| Speed Limit 60     							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares infavorably to the accuracy on the test set of 92.4%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10th through 12th cell of the Ipython notebook. However I had trouble running it in the notebook. The code is self contained in the script 'new_images_script.py', included in the submission.

For the first image, 'Road Work 25'
the model is correct and relatively sure that this is a stop sign (probability of .99), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Road Work  									| 
| 0.0016     				| Speed Limit 60 										|
| 0.00025					| Priority Road											|
| 0.00018	      			| Right of Way					 				|
| 0.000052				    | General Caution     							|


For the second image, 'Keep Right 38'
the model is incorrect and not very confident that this is a Speed Limit 60 (probability of .58), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.58         			| Speed Limit 60  									| 
| 0.35  | Priority Road	 										|
| 0.027					| Road Work 										|
| 0.018	      			| Keep Right					 				|
| 0.011				    | Right of Way at next intersection     							|

For the third image, 'Priority Road 12'
the model is correct and confident that this is a Priority Road 12 (probability of .0.79), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|  0.79	| Priority Road 									| 
| 0.12  | Road Work 	 										|
| 0.08					| Speed Limit 60 										|
| 0.0034	      			| Right of Way at next intersection					 				|
| 0.00028				    | Keep Right     							|

For the fourth image, 'dangerous curve left 19'
the model is incorrect and is oddly very confident that this is a Road Work 25 (probability of 0.89), and the image does contain a road work sign. The mistake is probably the shared strong triangle, white and red edge signs that both signages have. The content in the center is hard to differentiate at 32x32 resoluion. In fact other top fives share this sign shape similarity.
The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.89  |Road Work  									| 
| 0.067  | Priority Road	 										|
| 0.027					| Right of Way at next intersection 										|
| 0.008      			| General Caution					 				|
| 0.0025				    | Wild Animal Crossing    							|

For the fifth image, 'Speed Limit 60'
the model is correct and very confident that this is a Speed Limit 60 (probability of 0.82), and the image does contain a road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.82   | Speed Limit 60  									| 
| 0.14  | Road Work  										|
| 0.031					| Priority Road										|
| 0.0028	      			| Right of Way at next intersection						 				|
| 0.00013				    | Keep Right     							|
