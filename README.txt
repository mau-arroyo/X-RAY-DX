Machine Learning Model

Image classification is one of the most exiting parts of machine learning. The ability of computers to recognize patterns and objects from images is an incredibly powerful tool in our toolkit.  However, before we can apply machine learning to images we need to transform the raw images to features usable by our learning algorithms.
We used a dataset of 5856 images of thorax x-rays separated in two classifications, with and without pneumonia, or normal and pneumonia, which are the labels.
We extracted the dataset from our local directory and used the OpenCV library to begin the preprocessing. Fundamentally, images are data and when we used imread we convert that data into a NumPy array. 
OpenCV library allow us to load the image as grayscale, use thresh_binary to output a simplified version and resize the image to standardize our dataset and to reduce memory usage.
 
We declare as empty lists where our X and y for training will be, which are our images and categories respectively. We will reshape the images to determine the dimensions of our images.
 
We can see an example of our image in process, even our image transformed into a matrix whose elements correspond to individual pixels, and each one has it’s value displayed, from black (0) to white (255).
 

We used NumPy’s flatten to convert the multidimensional array containing an image’s data into a vector containing the observation’s values (10,000). Then normalize the data by rescaling the values to be between 0 and 1. And lastly we will transform y_train to categorical data.
 

We decided to use a convolutional neural network because they have proved to be very effective in the computer vision branch, and although is possible to use a feedforward NN the convolutional Network can take into account the special structure of the pixels and also are able to detect an object regardless of where it appears in an image.
Arquitecture
First, we add a
convolutional layer and specify the number of filters and other characteristics.
Second, we add a max pooling layer, summarizing the nearby pixels
Third, we add a dropout layer to
reduce the chances of overfitting.

Fourth, we add a flatten layer to convert the convolutionary
inputs into a format able to be used by a fully connected layer.
Finally, we
add the fully connected layers and an output layer to do the actual classification.
Notice that because this is a multiclass classification problem, we use the softmax
activation function in the output layer.
 

We trained our model only for 2 epochs after recollecting data from training and grid search and we determined it was the best moment to stop training to prevent overfitting. 
Earlystopping, added another layer with dropout and max pooling and this was the model with the best results also the simpler.
