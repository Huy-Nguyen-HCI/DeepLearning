import java.util.AbstractCollection;
import java.util.HashMap;
import Jama.Matrix;
import org.omg.PortableInterceptor.ACTIVE;

public class ConvolutionalNeuralNetwork {

	int lossFunctionType = LossFunction.LOG_LOSS;
	Layer[] layers;

	public ConvolutionalNeuralNetwork() {
		layers = new Layer[4];
		layers[0] = new ConvolutionalLayer( 2, 3, 2, 1, ActivationFunctions.LINEAR );
		layers[1] = new MaxPoolingLayer( 2, 1 );
		layers[2] = new FullyConnectedLayer( 5, ActivationFunctions.LINEAR );
		layers[3] = new FullyConnectedLayer( 3, ActivationFunctions.SOFTMAX );
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
					((ConvolutionalLayer) layers[0]).setFilters(
							new double[][][][] {
									new double[][][] {
											new double[][] {
													new double[]{1,1,1},
													new double[]{1,1,1},
													new double[]{1,1,1}
											},
											new double[][] {
													new double[]{0,-1,0},
													new double[]{0,1,1},
													new double[]{-1,1,-1}
											},
											new double[][] {
													new double[]{1,0,1},
													new double[]{-1,1,0},
													new double[]{-1,1,1}
											}
									},
									new double[][][] {
											new double[][] {
													new double[]{1,1,0},
													new double[]{0,0,-1},
													new double[]{0,0,1}
											},
											new double[][] {
													new double[]{-1,0,-1},
													new double[]{-1,1,-1},
													new double[]{-1,0,1}
											},
											new double[][] {
													new double[]{1,1,-1},
													new double[]{-1,1,1},
													new double[]{1,1,0}
											}
									}
							}

					);
				}
				convLayer.computeLinearCombinations();
				threeDimensionalInput = convLayer.computeOutput();
				Utilities.print3DMatrix( threeDimensionalInput );
			}
			else if ( layers[i] instanceof MaxPoolingLayer ) {
				MaxPoolingLayer maxPool = (MaxPoolingLayer) layers[i];
				maxPool.setInput( threeDimensionalInput );
				threeDimensionalInput = maxPool.computeOutput();
				Utilities.print3DMatrix( threeDimensionalInput );
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
				Utilities.printArray( oneDimensionalInput );
			}
		}
		Utilities.printArray( oneDimensionalInput );
	}


	public void backwardPropagation() {

	}

}