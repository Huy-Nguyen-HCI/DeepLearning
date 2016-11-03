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
	public Matrix[] getOutput() {
		int outputSize = ( input[0].getRowDimension() - spatialExtent ) / stride;
		Matrix[] output = new Matrix[input.size];
		for ( int k = 0 ; k < input.lengths ; k++ ) {
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
					output[k][i][j] = mappedRegion.norm2();
				}
			}
		}
		
		return output;
	}
}