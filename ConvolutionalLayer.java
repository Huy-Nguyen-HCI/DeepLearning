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
    Matrix[] delta;

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
			filters[k].computeGradient( input, delta, stride );
		}
	}


	public double getOutput( int depth, int x, int y ) {
		return ActivationFunctions.applyActivationFunction( activationFunction, linearCombinations[depth].get(x,y) );
	}


	public Matrix[] propagateError() {
        Matrix[] error = new Matrix[Filter.FILTER_DEPTH];
        for ( int depth = 0 ; depth < FILTER_DEPTH ; depth++ ) {
            Matrix errorSlice = error[depth];
            for ( int i = 0 ; i < input[depth].getRowDimension() ; i++ ) {
                for ( int j = 0 ; j < input[depth].getColumnDimension() ; j++ ) {
                    // calculate error at neuron (i,j) at input[depth]
                    double err = 0;
                    for ( int a = 0 ; a < filterSize ; a++ ) {
                        for ( int b = 0 ; b < filterSize ; b++ ) {
                            err += 0;
                        }
                    }
                }
            }
        }
        return error;
	}


	public void setError( Matrix[] error ) {
        // compute the delta for all nodes
        delta = new Matrix[error.length];
        for ( int k = 0 ; k < error.length ; k++ ) {
            for ( int i = 0 ; i < error[k].getRowDimension() ; i++ ) {
                for ( int j = 0 ; j < error[k].getColumnDimension() ; j++ ) {
                    double deltaAtNode = error[k].get(i,j) * linearCombinations[k].get(i,j);
                    delta[k].set( i, j, deltaAtNode );
                }
            }
        }
	}

}