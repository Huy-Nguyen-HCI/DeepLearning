import java.util.Arrays;
import java.util.Random;
import Jama.Matrix;

public class Main {

//    public static void main(String[] args) {
//        ConvolutionalNeuralNetwork network = new ConvolutionalNeuralNetwork();
//        double[][][] arrInput = new double[][][]{
//                new double[][]{
//                        new double[]{2, 0, 1, 1, 0},
//                        new double[]{2, 1, 2, 2, 1},
//                        new double[]{2, 0, 0, 1, 2},
//                        new double[]{1, 1, 2, 2, 1},
//                        new double[]{0, 1, 0, 2, 2}
//                },
//                new double[][]{
//                        new double[]{1, 2, 1, 1, 2},
//                        new double[]{1, 2, 1, 2, 0},
//                        new double[]{2, 0, 1, 2, 2},
//                        new double[]{2, 2, 2, 1, 0},
//                        new double[]{0, 1, 0, 2, 2}
//                },
//                new double[][]{
//                        new double[]{0, 0, 2, 0, 0},
//                        new double[]{1, 1, 1, 0, 2},
//                        new double[]{2, 1, 1, 2, 1},
//                        new double[]{0, 2, 1, 1, 0},
//                        new double[]{0, 0, 2, 1, 2}
//                }
//        };
//        Matrix[] input = new Matrix[3];
//        for (int i = 0; i < input.length; i++) {
//            input[i] = new Matrix(arrInput[i]);
//        }
//        network.forwardPropagation(input);
//        double[] target = new double[]{0, 1};
//        network.backwardPropagation(target);
//    }
//}


    public static void main(String[] args) {
        double[][] data = Utilities.readFile("mnist_train.csv");
        double[][] inputs = getPixels( data );
        double[][] targets = getLabels( data );

        ConvolutionalNeuralNetwork network = new ConvolutionalNeuralNetwork();
        ConvolutionalTraining training = new ConvolutionalTraining( network );

        training.stochasticTraining( inputs, targets );


        data = Utilities.readFile("mnist_test.csv");
        inputs = getPixels( data );
        targets = getLabels( data );
        training.test( inputs, targets );
    }


    public static double[][] getLabels( double[][] data ) {
        double[][] labels = new double[data.length][10];
        for ( int i = 0 ; i < data.length ; i++ ) {
            double correctAnswer = data[i][0];
            for ( int j = 0 ; j < labels[i].length ; j++ ) {
                labels[i][j] = ( j == (int)correctAnswer ) ? 1 : 0;
            }
        }
        return labels;
    }


    public static double[][] getPixels( double[][] data ) {
        double[][] pixels = new double[data.length][data[0].length - 1];
        for ( int i = 0 ; i < data.length ; i++ ) {
            for ( int j = 1 ; j < data[i].length ; j++ ) {
                pixels[i][j-1] = data[i][j];
            }
        }
        return pixels;
    }
}