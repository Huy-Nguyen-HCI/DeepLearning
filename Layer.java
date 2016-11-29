import Jama.Matrix;

/**
 * Class that represents a neuron layer in a neural network.
 */
public class Layer {

	public static final int
			CONVOLUTIONAL = 0,
			MAX_POOLING = 1,
			FULLY_CONNECTED = 2;

	Matrix[] input;
	int activationFunction;

	public Layer( int activationFunction ) {
		this.activationFunction = activationFunction;
	}

	public Layer() { this(-1); }

	public void setInput( Matrix[] input ) {
		this.input = input;
	}

	public double[] propagateOneDimensionalError() { return null; }

	public Matrix[] propagateThreeDimensionalError() { return null; }

	public void updateWeights() { }

	public void clearData() { input = null; }

}