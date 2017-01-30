package core;

import layer.*;
import convolution.ConvolutionalTraining;
import data.Dataset;

public class Main {

    public static void main( String[] args ) {
        // construct a conv network with two convolutional layers (followed by maxpooling)
        // and an output sigmoid layer
        Network network = new Network();
        network.addLayer( new InputLayer(new Size(28,28)) );
        network.addLayer( new ConvolutionalLayer(6, new Size(5,5)) );
        network.addLayer( new MaxpoolingLayer(new Size(2,2)) );
        network.addLayer( new ConvolutionalLayer(12, new Size(5,5)) );
        network.addLayer( new MaxpoolingLayer(new Size(2,2)) );
        network.addLayer( new OutputLayer(10) );
        ConvolutionalTraining training = new ConvolutionalTraining( network, 50 );

        // import the dataset
        Dataset dataset = Dataset.load( "data/mnist_train.csv", ",", 0 );
        Dataset testset = Dataset.load( "data/mnist_test.csv", ",", 0 );
        training.train( dataset, 100 );

        //  testing
        training.test( testset );
    }
}