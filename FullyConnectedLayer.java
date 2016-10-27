import Jama.Matrix;
/**
 * Class that represents a fully connected layer in a feedforward or convolution NN.
 */
public class FullyConnectedLayer extends Layer {

	public FullyConnectedLayer( int activationFunction ) {
		super( activationFunction );
	}
}