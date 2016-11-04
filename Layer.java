import Jama.Matrix;

/**
 * Class that represents a neuron layer in a neural network.
 */
public class Layer {

	Matrix input;
	int activationFunction;
	Matrix output;

	public Layer( int activationFunction ) {
		this.activationFunction = activationFunction;
	}

	
	public void setInput( Matrix input ) {
		this.input = input;
	}	


	public Matrix getOutput() {
		return (output == null) ? input : output;
	}
}