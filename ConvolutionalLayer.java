import Jama.Matrix;

/**
 * Class that represents a convolutional layer in a convolutional neural network.
 */
public class ConvolutionalLayer extends Layer {

	// filter parameters
	Matrix[] filters;
	int stride; // number of steps to take for next scan
	int filterSize; // size of filter

	// backpropagation info
	Matrix[] output;
	double[][] delta;
	double[][][] gradients;

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
		this.stride = stride;
		padInput( padding );
	}


	public void padInput( int padding ) {
		for ( int k = 0 ; k < input.length ; k++ ) {
			Matrix padded = new Matrix( input[k].getRowDimension() + padding , input[k].getColumnDimension() + padding );
			int m = padded.getRowDimension();
			int n = padded.getColumnDimension();
			for ( int i = 0 ; i < m ; i++ ) {
				for ( int j = 0 ; j < n ; j++ ) {
					padded.set( i, j, (i >= padding && i < m - padding) ? input[k].get(i-m,j-m) : 0);
				}
			}
			input[k] = padded;
		}		
	}


	public Matrix[] getOutput() {
		// each filter produces one 2D output
		int inputWidth = input[0].getRowDimension();
		int inputHeight = input[0].getColumnDimension();

		Matrix[] output = new Matrix[filters.length];
		int outputSize = inputWidth - filters[0].length + 1;

		for ( int k = 0 ; k < filters.length ; k++ ) {
			Matrix filter = filter[k];
			output[k] = new Matrix( outputSize, outputSize );
			for ( int i = filterSize / 2 ; i < inputWidth - filterSize / 2 ; i+= stride ) {
				for ( int j = filterSize / 2 ; j < inputHeight - filterSize / 2 ; j+= stride ) {
					Matrix mappedRegion = input[k].getMatrix( 
						i - filterSize / 2, 
						i + filterSize / 2, 
						j - filterSize / 2, 
						j + filterSize / 2 
					);
					double linearCombination = sum( mappedRegion.arrayTimes( filter ) );
					output[k][i][j] = ActivationFunctions.applyActivationFunction( activationFunction, linearCombination );
				}
			}
		}
		this.output = output;
		return output;
	}
	

	public void computeNodeDelta() {
		for ( int i = filterSize / 2 ; i <= inputWidth - filterSize / 2 ; i+= stride ) {
			for ( int j = filterSize / 2 ; j <= inputHeight - filterSize / 2 ; j+= stride ) {
				Matrix neuron = output.getMatrix( 
					i - filterSize / 2, 
					i + filterSize / 2, 
					j - filterSize / 2, 
					j + filterSize / 2 
				);
				
			}
		}
	}


	public void clearData() {
		output = null;
		delta = new Matrix( delta.getRowDimension(), delta.getColumnDimension() );
	}
}