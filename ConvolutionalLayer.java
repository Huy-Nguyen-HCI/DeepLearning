import Jama.Matrix;

/**
 * Class that represents a convolutional layer in a convolutional neural network.
 */
public class ConvolutionalLayer extends Layer {

	Matrix[] filters;
	int stride;
	int padding;
	int filterSize;
	Matrix[] output;

	/**
	* Take a 3D matrix as input.
	*/
	public ConvolutionalLayer(
		int numberOfFilters,
		int filterSize,
		int stride,
		int padding,
		int activationFunctionType
	) {
		super( activationFunctionType );

		this.filterSize = filterSize;
		// initialize the filters
		filters = new Matrix[filterSize];
		for ( int i = 0 ; i < numberOfFilters ; i++ ) {
			filters = new Matrix[filterSize][filterSize];
		}

		this.padding = padding;
		this.stride = stride;
	}


	public Matrix[] getOutput() {
		// each filter produces one 2D output
		Matrix[] output = new Matrix[filters.length];

		int outputSize = input.getRowDimension() - filters[0].length + 1;
		for ( int k = 0 ; k < output.length ; k++ ) {
			Matrix filter = filter[k];
			output[k] = new Matrix(outputSize,outputSize);
			for ( int i = 0 ; i < outputSize ; i++ ) {
				for ( int j = 0 ; j < outputSize ; j++ ) {
					Matrix mappedRegion = input[k].getMatrix( 
						i, 
						i + filterSize - 1, 
						j, 
						j + filterSize - 1 
					);
					double linearCombination = sum( mappedRegion.arrayTimes( filter ) );
					output[k][i][j] = ActivationFunctions.applyActivationFunction( activationFunction, linearCombination );
				}
			}
		}
		this.output = output;
		return output;
	}



	/**
	 * Return the padding p such that steps = (w - f + 2p) / (s+1) is an integer
	 * @param w width of the input image
	 * @return the value of p.
	 */
	private int getPadding( int w ) {
		return Math.abs(2p - f) % (s + 1);
	}


	/**
	 * Get the number of steps needed for each filter to scan through one row
	 * @param w width of the input image
	 * @return (w - f + 2p) / (s+1)
	 */
	public int getNumberOfSteps( int w ) {
		int p = getPadding( w );
		int f = filters[0].getRowDimension();
		return ( w - f + 2*p ) / (stride + 1);
	}
}