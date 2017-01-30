public class Main {

    public static void main( String[] args ) {
        // construct a conv network with two convolutional layers (followed by maxpooling)
        // and an output softmax layer
        Network network = new Network();
        network.addLayer( Layer.buildInputLayer(new Size(28,28)) );
        network.addLayer( Layer.buildConvLayer(6, new Size(5,5)) );
        network.addLayer( Layer.buildMaxpoolLayer(new Size(2,2)) );
        network.addLayer( Layer.buildConvLayer(12, new Size(5,5)) );
        network.addLayer( Layer.buildMaxpoolLayer(new Size(2,2)) );
        network.addLayer( Layer.buildOutputLayer(10) );
        ConvolutionalTraining training = new ConvolutionalTraining( network, 50 );

        // import the dataset
        Dataset dataset = Dataset.load( "mnist_train.csv", ",", 0 );
        Dataset testset = Dataset.load( "mnist_test.csv", ",", 0 );
        training.train( dataset, testset, 100 );

        //  testing
        training.test( testset );
    }
}