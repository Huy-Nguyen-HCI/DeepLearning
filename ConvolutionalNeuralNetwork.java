import java.util.AbstractCollection;
import java.util.HashMap;
import Jama.Matrix;
import org.omg.PortableInterceptor.ACTIVE;

public class ConvolutionalNeuralNetwork {

	int lossFunctionType = LossFunction.LOG_LOSS;
	Layer[] layers;

	public ConvolutionalNeuralNetwork() {
		layers = new Layer[3];
		layers[0] = new ConvolutionalLayer( 1, 2, 1, 0, ActivationFunctions.LINEAR );
		layers[1] = new FullyConnectedLayer( 2, ActivationFunctions.LINEAR );
		layers[2] = new FullyConnectedLayer( 2, ActivationFunctions.SOFTMAX );
	}


	public void forwardPropagation( Matrix[] input ) {
		Matrix[] threeDimensionalInput = input;
		Utilities.print3DMatrix( threeDimensionalInput );
		double[] oneDimensionalInput = null;
		for ( int i = 0 ; i < layers.length ; i++ ) {
			if ( layers[i] instanceof ConvolutionalLayer ) {
				ConvolutionalLayer convLayer = (ConvolutionalLayer) layers[i];
				convLayer.setInput( threeDimensionalInput );
				if ( i == 0 ) {
					setInitialWeightsForTesting();
				}
				convLayer.computeLinearCombinations();
				threeDimensionalInput = convLayer.computeOutput();
				Utilities.print3DMatrix( threeDimensionalInput );
			}
			else if ( layers[i] instanceof MaxPoolingLayer ) {
				MaxPoolingLayer maxPool = (MaxPoolingLayer) layers[i];
				maxPool.setInput( threeDimensionalInput );
				threeDimensionalInput = maxPool.computeOutput();
//				Utilities.print3DMatrix( threeDimensionalInput );
			}
			else {
				assert ( layers[i] instanceof FullyConnectedLayer );
				FullyConnectedLayer fullLayer = (FullyConnectedLayer) layers[i];
				Layer previousLayer = layers[i-1];
				if ( previousLayer instanceof ConvolutionalLayer|| previousLayer instanceof MaxPoolingLayer ) {
					// take a 3D matrix as input
					fullLayer.setInput( threeDimensionalInput );
				}
				else {
					// take a 1D vector as input
					assert ( oneDimensionalInput != null );
					fullLayer.setInput( oneDimensionalInput );
				}
				fullLayer.computeLinearCombinations();
				oneDimensionalInput = fullLayer.computeOutput();
//				Utilities.printArray( oneDimensionalInput );
			}
		}
		Utilities.printArray( oneDimensionalInput );
	}


	public void backwardPropagation( double[] target ) {
		System.out.println("\n\nTESTING BACKPROPAGATION\n\n");
		double[] oneDimensionalError = null;
		Matrix[] threeDimensionalError = null;
		for ( int i = layers.length - 1 ; i >= 0 ; i-- ) {

			// calculate deltas and gradients
			if ( i == layers.length - 1 ) {
				assert( layers[i] instanceof FullyConnectedLayer );
				FullyConnectedLayer outputLayer = (FullyConnectedLayer) layers[i];
				outputLayer.computeNodeDeltasForOutputLayer( target, LossFunction.LOG_LOSS );
				outputLayer.computeGradients();
			}
			else {
				if ( layers[i] instanceof FullyConnectedLayer ) {
					assert( oneDimensionalError != null );
					FullyConnectedLayer fullLayer = (FullyConnectedLayer) layers[i];
					fullLayer.setErrorAndComputeDeltas( oneDimensionalError );
					fullLayer.computeGradients();
				}
				else if ( layers[i] instanceof MaxPoolingLayer ) {
					assert( threeDimensionalError != null );
					MaxPoolingLayer maxPool = (MaxPoolingLayer) layers[i];
					maxPool.setError( threeDimensionalError );
				}
				else {
					assert( threeDimensionalError != null );
					ConvolutionalLayer convLayer = (ConvolutionalLayer) layers[i];
					convLayer.setErrorAndComputeDeltas( threeDimensionalError );
					convLayer.computeGradients();
				}
			}

			// propagate errors
			if ( i > 0 && layers[i-1] instanceof FullyConnectedLayer ) {
				oneDimensionalError = layers[i].propagateOneDimensionalError();
			}
			else {
				threeDimensionalError = layers[i].propagateThreeDimensionalError();
			}
		}

		for ( int i = layers.length - 1 ; i >= 0 ; i-- ) {
			if ( layers[i] instanceof FullyConnectedLayer ) {
				System.out.println("delta at layer " + i + ":");
				Utilities.printArray( ((FullyConnectedLayer) layers[i]).delta );
				System.out.println("gradients at layer " + i + ":");
				for ( double[] mat : ((FullyConnectedLayer) layers[i]).gradients ) {
					Utilities.printArray( mat );
					System.out.println("&&&");
				}

				System.out.println();
			}
			else if ( layers[i] instanceof ConvolutionalLayer ) {
				System.out.println("delta at layer " + i + ":");
				Utilities.print3DMatrix( ((ConvolutionalLayer) layers[i]).delta );
				System.out.println("gradients at layer " + i + ":");
				((ConvolutionalLayer) layers[i]).printGradients();
				System.out.println();
			}
		}
	}


	private void setInitialWeightsForTesting() {
		((ConvolutionalLayer) layers[0]).setFilters(
				new double[][][][] {
						new double[][][] {
								new double[][] {
										new double[]{1,0},
										new double[]{0,1},
								},
								new double[][] {
										new double[]{1,1},
										new double[]{0,0},
								},
								new double[][] {
										new double[]{0,1},
										new double[]{1,0},
								}
						}
				}
		);
	}

}