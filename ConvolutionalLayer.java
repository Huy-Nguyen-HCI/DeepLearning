import Jama.Matrix;

import javax.swing.*;

/**
 * Class that represents a convolutional layer in a convolutional neural network.
 */
public class ConvolutionalLayer extends Layer {

	// filter has the same depth with images
	final int FILTER_DEPTH = 3;
	
	// filter parameters
	Filter[] filters;
	int stride; // number of steps to take for next scan
	int filterSize; // width and height of filter

	// backpropagation info
	Matrix[] linearCombinations;
    Matrix[] delta;
	double[] bias;

	/**
	* Take a 3D matrix as input.
	*/
	public ConvolutionalLayer(
			int numberOfFilters,
			int filterSize,
			int stride,
			int padding,
			int activationFunctionType,
			double[] bias)
	{
		super( activationFunctionType );
		this.filterSize = filterSize;
		filters = new Filter[numberOfFilters];
		linearCombinations = new Matrix[numberOfFilters];
		this.stride = stride;
		padInput( padding );
		assert ( bias.length == numberOfFilters );
		this.bias = bias;
	}


	/**
	 * Pad the border of the input with 0s. If input has dimension K x K x 3 then after paddding.
	 * it becomes (K + padding) x (K + padding) x 3
	 * @param padding the number of layers of 0s to pad
	 */
	public void padInput( int padding ) {
		assert input.length == Filter.FILTER_DEPTH;
		for ( int k = 0 ; k < Filter.FILTER_DEPTH ; k++ ) {
			Matrix padded = new Matrix( input[k].getRowDimension() + padding , input[k].getColumnDimension() + padding );
			int m = padded.getRowDimension();
			int n = padded.getColumnDimension();
			for ( int i = 0 ; i < m ; i++ ) {
				for ( int j = 0 ; j < n ; j++ ) {
					double newValue = (i >= padding && i < m - padding) ? input[k].get(i-padding,j-padding) : 0;
					padded.set( i, j, newValue );
				}
			}
			input[k] = padded;
		}		
	}


	/**
	 * Calculate all the 2D boards resulting from scanning a filter slice on an image slice.
	 */
	public void computeLinearCombinations() {
		// each filter produces one 2D output
		for ( int k = 0 ; k < filters.length ; k++ ) {
			linearCombinations[k] = filters[k].computeLinearCombination( input, stride, bias[k] );
		}
	}


	/**
	 * Compute the gradient of each weight in the 4D weight matrix (the filters)
	 */
	public void computeGradients() {
		for ( int k = 0 ; k < filters.length ; k++ ) {
			filters[k].computeGradient( input, delta, stride );
		}
	}


	/**
	 * Get the output matrix.
	 * @return the linear combination matrix after applying activation functions
	 */
	public Matrix[] getOutput() {
		Matrix[] output = new Matrix[linearCombinations.length];
		for ( int k = 0 ; k < linearCombinations ; k++ ) {
			output[k] = new Matrix( linearCombinations[k].getRowDimension(), linearCombinations[k].getColumnDimension() );
			for ( int i = 0 ; i < linearCombinations[k].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < linearCombinations[k].getColumnDimension() ; j++ ) {
					output[k].set( i, j, getOutput(k, i, j) );
				}
			}
		}
		return output;
	}


	/**
	 * Get the output value at the specified location
	 * @param depth the index of the 2D slice
	 * @param x the x-value within the 2D slice
	 * @param y the y-value within the 2D slice
	 * @return the number at location <tt>output[depth][x][y]</tt>
	 */
	public double getOutput( int depth, int x, int y ) {
		return ActivationFunctions.applyActivationFunction( activationFunction, linearCombinations[depth].get(x,y) );
	}


	/**
	 *
	 * @return
	 */
	public Matrix[] propagateError() {
        Matrix[] error = Utilities.createMatrixWithSameDimension( input );
        for ( int depth = 0 ; depth < Filter.FILTER_DEPTH ; depth++ ) {
			for ( int i = )
		}
        return error;
	}


	public void setError( Matrix[] error ) {
        // compute the delta for all nodes
        delta = Utilities.createMatrixWithSameDimension( error );
        for ( int k = 0 ; k < error.length ; k++ ) {
            for ( int i = 0 ; i < error[k].getRowDimension() ; i++ ) {
                for ( int j = 0 ; j < error[k].getColumnDimension() ; j++ ) {
                    double deltaAtNode = error[k].get(i,j) *
							ActivationFunctions.applyActivationFunctionDerivative( activationFunction, linearCombinations[k].get(i,j) );
                    delta[k].set( i, j, deltaAtNode );
                }
            }
        }
	}

}