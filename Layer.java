import Jama.Matrix;
import Jama.Matrix;

/**
 * Class that represents a neuron layer in a neural network.
 */
public class Layer {

	Matrix input;
	int activationFunction;

	public Layer( int activationFunction ) {
		this.activationFunction = activationFunction;
	}

	
	public void setInput( Matrix input ) {
		this.input = input;
	}	


	public Matrix getOutput() {
		return input;
	}
}