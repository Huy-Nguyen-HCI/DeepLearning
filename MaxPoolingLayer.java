import Jama.Matrix;

/**
 * Class that represents a maxpooling layer in a convolutional network.
 */
public class MaxPoolingLayer extends Layer {

	int spatialExtent;
	int stride;

	public MaxPoolingLayer( int spatialExtent, int stride, int activationFunction ) {
		super( activationFunction );
		this.spatialExtent = spatialExtent;
		this.stride = stride;
	}

	/**
	 * Return a 3D matrix of smaller dimension after maxpooling
	 */
	public Matrix getOutput() {
		return null;
	}
}