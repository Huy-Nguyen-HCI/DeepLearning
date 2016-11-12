import Jama.Matrix;
/**
 * Class that represents a fully connected layer in a feedforward or convolution NN.
 */
public class FullyConnectedLayer extends Layer {

	int neuronCount;
	Matrix[][] weights;

	public FullyConnectedLayer( int neuronCount, int activationFunction ) {
		super( activationFunction );
		this.neuronCount = neuronCount;
	}

//	@Override
//	public void setInput( Matrix[] input ) {
//		this.input = input;
//		// initialize weight array
//		int inputSize = input[0].getRowDimension();
//		weights = new double[neuronCount][][][];
//		for ( int i = 0 ; i < weights.length ; i++ ) {
//			weights[i] = new double[input.length][][];
//			for ( int j = 0 ; j < weights[i].length ; j++ ) {
//				weights[i][j] = new Matrix( inputSize, inputSize );
//			}
//		}
//	}


	public void computeLinearCombinations() {

	}


//	public double output() {
//
//	}
}