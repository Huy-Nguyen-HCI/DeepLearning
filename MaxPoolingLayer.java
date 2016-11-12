import Jama.Matrix;

/**
 * Class that represents a maxpooling layer in a convolutional network.
 */
public class MaxPoolingLayer extends Layer {

	int spatialExtent;
	int stride;
	Matrix[] error;

	public MaxPoolingLayer( int spatialExtent, int stride, int activationFunction ) {
		super( activationFunction );
		this.spatialExtent = spatialExtent;
		this.stride = stride;
	}

	/**
	 * Return a 3D matrix of smaller dimension after maxpooling
	 */
	public Matrix[] getOutput() {
		Matrix[] output = new Matrix[input.length];
		for ( int k = 0 ; k < input.length ; k++ ) {
			int numberOfSteps = ( input[k].getRowDimension() - spatialExtent ) / stride + 1;
			output[k] = new Matrix( numberOfSteps , numberOfSteps );
			// downsample input[k] to output[k]
			for ( int i = 0 ; i < numberOfSteps ; i++ ) {
				for ( int j = 0 ; j < numberOfSteps ; j++ ) {
					// look at a 2D board at depth k
					Matrix mappedRegion = input[k].getMatrix(
							stride * i,
							stride * i + spatialExtent - 1,
							stride * j,
							stride * j + spatialExtent - 1
					);
					double max = mappedRegion.findMax();
					output[k].set( i, j, max );
				}
			}
		}
		return output;
	}


	public void setError( Matrix[] error ) {
		this.error = error;
	}


	public Matrix[] propagateError() {
		return error;
	}
}