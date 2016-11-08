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
		int outputSize = ( input[0].getRowDimension() - spatialExtent ) / stride;
		assert( input.length == Filter.FILTER_DEPTH );
		Matrix[] output = new Matrix[input.length];
		for ( int k = 0 ; k < output.length ; k++ ) {
			output[k] = new Matrix( outputSize, outputSize );
			for ( int i = 0 ; i < outputSize ; i++ ) {
				for ( int j = 0 ; j < outputSize ; j++ ) {
					Matrix mappedRegion = input[k].getMatrix( 
						i * stride, 
						i * stride + spatialExtent - 1, 
						j * stride, 
						j * stride + spatialExtent - 1 
					);
					// get the maximum value
					output[k].set( i, j, mappedRegion.norm2() );
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