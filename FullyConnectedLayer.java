import Jama.Matrix;
/**
 * Class that represents a fully connected layer in a feedforward or convolution NN.
 */
public class FullyConnectedLayer extends Layer {

	int neuronCount;

	public FullyConnectedLayer( int neuronCount, int activationFunction ) {
		super( activationFunction );
		this.neuronCount = neuronCount;
	}


	public double output() {
		
	}
}