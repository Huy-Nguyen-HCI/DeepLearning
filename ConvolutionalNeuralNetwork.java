import java.util.HashMap;
import Jama.Matrix;

public class ConvolutionalNeuralNetwork {

	public static final int 
		CONVOLUTIONAL = 0,
		MAX_POOLING = 1,
		FULLY_CONNECTED = 2;

	int lossFunctionType = LossFunction.MSE;
	Layer[] layers;

//	 public ConvolutionalNeuralNetwork( int[] layerTypes, int[] activationFunctionTypes, double[] bias )
//	 {
//	 	// initialize the layers
//	 	layers = new Layer[ layerTypes.length ];
//	 	for ( int i = 0 ; i < layers.length ; i++ ) {
//	 		switch layerTypes[i] {
//	 			case CONVOLUTIONAL:
//	 				layers[i] = new ConvolutionalLayer()
//	 		}
//	 	}
//	 }


	public void forwardPropagation() {

	}


	public void backwardPropagation() {

	}

}