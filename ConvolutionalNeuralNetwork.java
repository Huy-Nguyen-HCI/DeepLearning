import java.util.HashMap;
import Jama.Matrix;

public class ConvolutionalNeuralNetwork {

	public static final int 
		CONVOLUTIONAL = 0,
		MAX_POOLING = 1,
		FULLY_CONNECTED = 2;

	int lossFunctionType = LossFunction.MSE;
	Layer[] layers;

	// public ConvolutionalNeuralNetwork( int[] layerTypes, int[] activationFunctionTypes, double[] bias ) 
	// {
	// 	// initialize the layers
	// 	layers = new Layer[ layerTypes.length ];
	// 	for ( int i = 0 ; i < layers.length ; i++ ) {
	// 		switch layerTypes[i] {
	// 			case CONVOLUTIONAL:
	// 				layers[i] = new ConvolutionalLayer()
	// 		}
	// 	}
	// }


	public void forwardPropagation() {

	}


	public void backwardPropagation() {

	}

	// public void computeNodeDeltaAtConvolutionalLayer(  ) {
	// 	Neuron node = network.getNode( network.getNumberOfLayers() - 1, nodeIndex );
	// 	switch (lossFunctionType) {
	// 		case MSE:
	// 			node.setDelta( 
	// 				(targets[nodeIndex] - node.output()) * node.getAFDerivative() 
	// 			);				
	// 			break;
	// 		case CROSS_ENTROPY:
	// 			node.setDelta( targets[nodeIndex] - node.output() );
	// 			break;
	// 		default:
	// 			assert false : "Error. Unrecognized loss function.";
	// 	}
	// }
}