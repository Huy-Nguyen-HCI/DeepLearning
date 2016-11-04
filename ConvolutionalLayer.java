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
	Matrix[] linearCombinations;
	Matrix[] gradients;
	Matrix[] error;

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
		// initialize the matrices
		filters = new Matrix[filterSize];
		for ( int i = 0 ; i < numberOfFilters ; i++ ) {
			filters = new Matrix[filterSize][filterSize];
		}

		linearCombinations = new Matrix[filters.length];
		gradients = new Matrix[filers.length];

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
					double newValue = (i >= padding && i < m - padding) ? input[k].get(i-m,j-m) : 0;
					padded.set( i, j, newValue );
				}
			}
			input[k] = padded;
		}		
	}


	public void computeLinearCombinations() {
		// each filter produces one 2D output
		int inputWidth = input[0].getRowDimension();
		int inputHeight = input[0].getColumnDimension();
		int outputSize = (inputWidth - filters[0].length) / stride + 1;
		for ( int k = 0 ; k < filters.length ; k++ ) {
			Matrix filter = filter[k];
			linearCombinations[k] = new Matrix( outputSize, outputSize );
			for ( int i = filterSize / 2 ; i < inputWidth - filterSize / 2 ; i+= stride ) {
				for ( int j = filterSize / 2 ; j < inputHeight - filterSize / 2 ; j+= stride ) {
					Matrix mappedRegion = input[k].getMatrix( 
						i - filterSize / 2, 
						i + filterSize / 2, 
						j - filterSize / 2, 
						j + filterSize / 2 
					);
					linearCombinations[k][i][j] = sum( mappedRegion.arrayTimes( filter ) );
				}
			}
		}
	}


	public void computeGradients() {
		for ( int k = 0 ; k < filters.length ; k++ ) {
			for ( int a = 0 ; a < filters[k].getRowDimension() ; a++ ) {
				for ( int b = 0 ; b < filters[k].getColumnDimension() ; b++ ) {					
					Matrix filter = filters[k];
					Matrix inputSlice = input[k];
					// weight at [a,b] connects to all neurons at (a+i*stride,b+j*stride)
					for ( int i = 0 ; a + i*stride < input[k].getRowDimension() ; i++ ) {
						for ( int j = 0 ; b + j*stride < input[k].getColumnDimension() ; j++ ) {
							gradients[k].get(i,j) += computeNodeDelta(k,i,j) * input[k].get(i+a,j+b);
						}
					}
				}
			}
		}
	}

	public double computeNodeDelta( int depth, int x, int y ) {
		return error[depth].get(x,y) * ActivationFunctions.applyActivationFunctionDerivative();
	}


	public double getOutput( int depth, int x, int y ) {
		return ActivationFunctions.applyActivationFunction( linearCombinations[depth].get(x,y) );
	}


	public Matrix[] propagateError() {
		return null;
	}


	public double getError( Matrix[] error ) {
		this.error = error;
	}


	public void clearData() {
		delta = new Matrix( delta.getRowDimension(), delta.getColumnDimension() );
	}
}