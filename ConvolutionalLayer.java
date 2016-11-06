import Jama.Matrix;

/**
 * Class that represents a convolutional layer in a convolutional neural network.
 */
public class ConvolutionalLayer extends Layer {
	
	final int FILTER_DEPTH = 3;
	
	// filter parameters
	Filter[] filters;
	int stride; // number of steps to take for next scan
	int filterSize; // size of filter

	// backpropagation info
	Matrix[] linearCombinations;
	Matrix[] error;

	/**
	* Take a 3D matrix as input.
	*/
	public ConvolutionalLayer(int numberOfFilters, int filterSize, int stride, int padding, int activationFunctionType){
		super( activationFunctionType );
		this.filterSize = filterSize;
		filters = new Filter[numberOfFilters];
		linearCombinations = new Matrix[numberOfFilters];
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
		for ( int k = 0 ; k < filters.length ; k++ ) {
			Filter filter = filters[k];
			linearCombinations[k] = filter.computeLinearCombination( input, stride );
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