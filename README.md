## MNIST digit classification task using PyTorch

Author: Kyuhong Shim(skhu20@snu.ac.kr)
If you have any questions on the code or README, please feel free to contact me.

### Performance

MLP: 98.19% (test accuracy) 

CNN: 99.41% (test accuracy)

### Description

The code provides predefined two different models.
The first model is MLP (multi layer perceptron), and he second model is CNN (convolutional neural networks).
The network is trained and evaluated on MNIST dataset with classification accuracy. 

#### Part 1: data preparation

The code use 'mnist_all.mat' file from Sam Roweis.
We split training data (60000) to train set (50000) and valid set (10000).
Test data (10000) is already given.
If the data is for MLP, data is 2-dim matrix, and for CNN, data is 4-dim tensor.

Mini-batch is first rescaled from [0, 255] to [0, 1].
The result is then normalized by precomputed mean and std.

#### Part 2: MLP model

The MLP model contains two hidden layers with ReLU activation function.
Each hidden layer is 256 dimension.
Hidden layer is initialized with default initialization.
Dropout of keep probability 0.5 is used for regularization.

#### Part 3: CNN model

The CNN model contains two convolutional layers with max pooling and ReLU function.
Each convolutional layer has 16 and 32 maps, respectively.
Both uses 5x5 kernels.
Dropout of keep probability 0.5 is used for regularization.
After convolution, the activation is flattened and passes through a fully connected layer.
Fully connected layer has 256 dimension.

#### Part 4: Training

For every epoch, we train network by train set and evaluate by valid set to check overfitting.
Patience increase when valid loss does not decrease.
Early stopping is done when the patience is bigger than max patience.
When early stopping appears, we multiply 0.1 to learning rate and restart the training with best model.
The test set is evaluated with best model with minimum valid loss.
