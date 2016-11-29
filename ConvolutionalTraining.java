/**
 * Created by nguyenha on 11/29/2016.
 */
public class ConvolutionalTraining {

    ConvolutionalNeuralNetwork network;
    final int numberOfIterations = 1000;
    int batchSize = 100;


    public ConvolutionalTraining( ConvolutionalNeuralNetwork network ) {
        this.network = network;
    }


    public void stochasticTraining( double[][] inputs, double[][] targets ) {
        for ( int j = 0 ; j < numberOfIterations ; j++ ) {
            System.out.println("&&& ITERATION " + j);
            int[] indices = Utilities.generateRandomNumbers( 0, inputs.length, batchSize );
            for ( int i = 0 ; i < indices.length ; i++ ) {
                iterate( i, inputs[indices[i]], targets[indices[i]] );
            }
            network.updateWeights();
        }
    }


    public void iterate( int iterationNumber, double[] inputs, double[] targets ) {
        network.forwardPropagation( Utilities.convert1DTo3D(inputs) );
        network.backwardPropagation( targets );
    }


    public void test( double[][] inputs, double[][] targets ) {
        double[] compares = new double[inputs.length];
        assert( inputs.length == targets.length );
        for ( int i = 0 ; i < inputs.length ; i++ ) {
            double[] outputs = network.forwardPropagation( Utilities.convert1DTo3D(inputs[i]) );
            int selectedChoiceIndex = Utilities.findIndexOfMax( outputs );
            int targetChoice = Utilities.findIndexOfMax( targets[i] );
            compares[i] = (selectedChoiceIndex == targetChoice) ? 1 : 0;
        }
        System.out.println("Accuracy is: ");
        double sum = 0 ;
        for ( double c : compares ) sum += c;
        System.out.println( sum / compares.length );
    }

}
