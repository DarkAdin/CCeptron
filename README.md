# CCeptron

A multi-layer perceptron written in C.

The entire code is a single C file. The program admits CSV files as data and all the important hyperparameters as arguments from the command line. In order to compile and use the program:

```{language=bash}
make
./CCeptron file.csv input_size hidden_size hidden_size2 output_size epochs learning_rate annealing_rate
```

The network comes with two hidden layers and one output layer by default. The two hidden layers use *tanh* as the activation function and the output layer uses *sigmoid*.

The addition and modification of any aspect of the network should be easy enough, whether you want to add or remove hidden layers, and/or different activation functions.

Once all training epochs have passed, the network tests itself on the training data. This, of course, should be done on a separate testing dataset.

The random number generator is seeded with the current time and the process ID.

In every iteration, CCeptron trains on a randomly picked row of the training data set, and does the same in every iteration of the testing stage, so it does not need to shuffle the training data and/or the testing data.

## Example of valid data

The following example row comes from the *iris* dataset (encoded accordingly):

```
0.645569,0.795454,0.202898,0.08,1.00
```

Which consists of

```
sepal_length,sepal_width,petal_length,petal_width,class
```

The class is encoded in the form of a number as well. If we have 3 different species in the *iris* dataset, an example of encoding them could be *0.0*, *0.5* and *1.0*. The maximum number of decimals in a data point should be 6.

## Inspirations

CCeptron is a minimalistic approach to a simple neural network concept, the perceptron. As such, it is not a fully capable neural network. But it should be easily modifiable to suit general needs and make predictions on small-to-medium complexity data. Check out these amazing machine learning projects in C which heavily inspired CCeptron:

* [tinn](https://github.com/glouw/tinn): tiny neural network written in C with one hidden layer.
* [darknet](https://github.com/glouw/tinn): one of the biggest machine learning projects in C, works with CUDA and heavily influenced the Computer Vision field, capable of almost anything.
* [genann](https://github.com/codeplea/genann)
* [kann](https://github.com/attractivechaos/kann): very complete project for constructing small to medium neural networks.

## TO-DO

These things will be added in the future:

* Set a way to read a separate testing set in order to do the testing
* Save all weights and biases of the network to make future predictions
* Multi-threading
