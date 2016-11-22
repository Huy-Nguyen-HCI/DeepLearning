import java.util.Arrays;
import java.util.Random;
import Jama.Matrix;

public class Main {

    public static void main(String[] args) {
        ConvolutionalNeuralNetwork network = new ConvolutionalNeuralNetwork();
        double[][][] arrInput = new double[][][]{
                new double[][]{
                        new double[]{1,1,1},
                        new double[]{2,2,2},
                        new double[]{3,3,3}
                },
                new double[][]{
                        new double[]{1,2,3},
                        new double[]{4,5,6},
                        new double[]{7,8,9}
                },
                new double[][]{
                        new double[]{0,0,0},
                        new double[]{1, 1, 1},
                        new double[]{2, 2, 2},
                }
        };
        Matrix[] input = new Matrix[3];
        for (int i = 0; i < input.length; i++) {
            input[i] = new Matrix(arrInput[i]);
        }
        network.forwardPropagation(input);
		double[] target = new double[]{ 0, 0, 1};
		network.backwardPropagation( target );
	}


//    public static void main( String[] args ) {
//        ConvolutionalLayer layer = new ConvolutionalLayer(2,3,2,1,ActivationFunctions.LINEAR );
//        double[][][] arrInput = new double[][][]{
//                new double[][]{
//                        new double[]{2,0,1,1,0},
//                        new double[]{2,1,2,2,1},
//                        new double[]{2,0,0,1,2},
//                        new double[]{1,1,2,2,1},
//                        new double[]{0,1,0,2,2}
//                },
//                new double[][]{
//                        new double[]{1,2,1,1,2},
//                        new double[]{1,2,1,2,0},
//                        new double[]{2,0,1,2,2},
//                        new double[]{2,2,2,1,0},
//                        new double[]{0,1,0,2,2}
//                },
//                new double[][]{
//                        new double[]{0,0,2,0,0},
//                        new double[]{1,1,1,0,2},
//                        new double[]{2,1,1,2,1},
//                        new double[]{0,2,1,1,0},
//                        new double[]{0,0,2,1,2}
//                }
//        };
//        Matrix[] input = new Matrix[3];
//        for ( int i = 0 ; i < input.length ; i++ ) {
//            input[i] = new Matrix( arrInput[i] );
//        }
//        layer.setInput( input );
//        layer.setFilters(
//                new double[][][][] {
//                        new double[][][] {
//                                new double[][] {
//                                        new double[]{1,1,1},
//                                        new double[]{1,1,1},
//                                        new double[]{1,1,1}
//                                },
//                                new double[][] {
//                                        new double[]{0,-1,0},
//                                        new double[]{0,1,1},
//                                        new double[]{-1,1,-1}
//                                },
//                                new double[][] {
//                                        new double[]{1,0,1},
//                                        new double[]{-1,1,0},
//                                        new double[]{-1,1,1}
//                                }
//                        },
//                        new double[][][] {
//                                new double[][] {
//                                        new double[]{1,1,0},
//                                        new double[]{0,0,-1},
//                                        new double[]{0,0,1}
//                                },
//                                new double[][] {
//                                        new double[]{-1,0,-1},
//                                        new double[]{-1,1,-1},
//                                        new double[]{-1,0,1}
//                                },
//                                new double[][] {
//                                        new double[]{1,1,-1},
//                                        new double[]{-1,1,1},
//                                        new double[]{1,1,0}
//                                }
//                        }
//                }
//
//        );
//        layer.computeLinearCombinations();
//        MaxPoolingLayer maxPool = new MaxPoolingLayer( 2, 1 );
//        maxPool.setInput( layer.computeOutput() );
//        Utilities.print3DMatrix( maxPool.computeOutput() );
//    }
}