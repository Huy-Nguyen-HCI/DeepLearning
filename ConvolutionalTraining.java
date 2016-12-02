/**
 * Created by nguyenha on 11/29/2016.
 */
public class ConvolutionalTraining {

    ConvolutionalNeuralNetwork network;
    final int numberOfIterations = 500;
    int batchSize = 100;


    public ConvolutionalTraining( ConvolutionalNeuralNetwork network ) {
        this.network = network;
    }


//    public void stochasticTraining( double[][] inputs, double[][] targets ) {
//        double[] error = new double[batchSize];
//        for ( int j = 0 ; j < numberOfIterations ; j++ ) {
//            System.out.println("&&& ITERATION " + j);
//            int[] indices = Utilities.generateRandomNumbers( 0, inputs.length, batchSize );
//            for ( int i = 0 ; i < indices.length ; i++ ) {
//                error[i] = iterate( i, inputs[indices[i]], targets[indices[i]] );
//            }
//            network.updateWeights( batchSize );
//            System.out.println("error at iteration " + j);
//            System.out.println( Utilities.findAverage(error) );
//        }
//    }


    public void stochasticTraining( double[][] inputs, double[][] targets ) {
        iterate( 0, inputs[0], targets[0] );
        iterate( 0, inputs[0], targets[0] );
    }


    public double iterate( int iterationNumber, double[] inputs, double[] targets ) {
        double[] outputs = network.forwardPropagation( Utilities.convert1DTo3D(inputs, 28) );
        Utilities.printArray( outputs );
//        network.backwardPropagation( targets );
//        return LossFunction.crossEntropyError( outputs, targets );
        return 0;
    }


    public void test( double[][] inputs, double[][] targets ) {
        double[] compares = new double[inputs.length];
        assert( inputs.length == targets.length );
        for ( int i = 0 ; i < inputs.length ; i++ ) {
            double[] outputs = network.forwardPropagation( Utilities.convert1DTo3D(inputs[i], 28) );
            System.out.println("output is:");
            Utilities.printArray( outputs );
            System.out.println();
            int selectedChoiceIndex = Utilities.findIndexOfMax( outputs );
            int targetChoice = Utilities.findIndexOfMax( targets[i] );
            compares[i] = (selectedChoiceIndex == targetChoice) ? 1 : 0;
        }
        System.out.println("Accuracy is: ");
        System.out.println( Utilities.getMean(compares) );
    }

}
