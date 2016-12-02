import Jama.Matrix;
import java.util.ArrayList;
import javax.swing.*;

/**
 * Class that represents a convolutional layer in a convolutional neural network.
 */
public class ConvolutionalLayer extends Layer {

	int padding;
	Filter[] filters; // filter parameters
	int stride; // number of steps to take for next scan
	int filterSize; // width and height of filter

	// backpropagation info
	Matrix[] linearCombinations;
    Matrix[] delta;

	/**
	* Take a 3D matrix as input.
	*/
	public ConvolutionalLayer(
			int numberOfFilters,
			int filterSize,
			int stride,
			int padding,
			int activationFunctionType)
	{
		super( activationFunctionType );
		this.filterSize = filterSize;
		filters = new Filter[numberOfFilters];
		linearCombinations = new Matrix[numberOfFilters];
		this.stride = stride;
		this.padding = padding;
	}

	/*********************** FEEDFORWARD ***************************************/

	@Override
	public void setInput( Matrix[] input ) {
		this.input = input;
		padInput( padding );
		if ( filters[0] == null ) {
			for ( int i = 0 ; i < filters.length ; i++ ) {
				filters[i] = new Filter( filterSize, input.length );
			}
		}
	}


	/**
	 * Pad the border of the input with 0s. If input has dimension K x K x 3 then after paddding.
	 * it becomes (K + padding) x (K + padding) x 3
	 * @param padding the number of layers of 0s to pad
	 */
	public void padInput( int padding ) {
		for ( int k = 0 ; k < input.length ; k++ ) {
			Matrix padded = new Matrix( input[k].getRowDimension() + 2*padding , input[k].getColumnDimension() + 2*padding );
			int m = padded.getRowDimension();
			int n = padded.getColumnDimension();
			for ( int i = 0 ; i < m ; i++ ) {
				for ( int j = 0 ; j < n ; j++ ) {
					double newValue = (i >= padding && i < m - padding && j >= padding && j < n - padding ) ? input[k].get(i-padding,j-padding) : 0;
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
			linearCombinations[k] = filters[k].computeLinearCombination( input, stride );
		}
	}


	/**
	 * Get the output matrix.
	 * @return the linear combination matrix after applying activation functions
	 */
	public Matrix[] computeOutput() {
		Matrix[] output = Utilities.createMatrixWithSameDimension( linearCombinations );
		for ( int k = 0 ; k < linearCombinations.length ; k++ ) {
			for ( int i = 0 ; i < linearCombinations[k].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < linearCombinations[k].getColumnDimension() ; j++ ) {
					double outputAtNeuron = ActivationFunctions.applyActivationFunction( activationFunction, linearCombinations[k].get(i,j) );
					output[k].set( i, j, outputAtNeuron );
				}
			}
		}
		return output;
	}


	/*********************** BACKPROPAGATION ***************************************/

	public void setErrorAndComputeDeltas( Matrix[] error ) {
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


	@Override
	public Matrix[] propagateThreeDimensionalError() {
		Matrix[] error = new Matrix[input.length];
		for ( int i = 0 ; i < error.length ; i++ ) {
			error[i] = new Matrix( input[i].getRowDimension() - 2*padding, input[i].getColumnDimension() - 2*padding );
		}
		for ( int k = 0 ; k < input.length ; k++ ) {
			for ( int i = padding ; i < input[k].getRowDimension() - padding ; i++ ) {
				for ( int j = padding ; j < input[k].getColumnDimension() - padding ; j++ ) {
					error[k].set( i - padding, j - padding, propagateErrorAtNeuron(k, i, j) );
				}
			}
		}
		return error;
	}


	private double propagateErrorAtNeuron( int depth, int row, int column ) {
		double error = 0;
		int outputSize = (input[0].getRowDimension() - filterSize) / stride + 1;
		for ( int a = 0 ; a < filterSize ; a++ ) {
			for ( int b = 0 ; b < filterSize ; b++ ) {
				int outputRow = (row - a) / stride;
				int outputColumn = (column - b) / stride;
				if ( outputRow >= 0 && outputRow < outputSize && outputColumn >= 0 && outputColumn < outputSize ) {
					for ( int filterIndex = 0 ; filterIndex < filters.length ; filterIndex ++ ) {
						error += delta[filterIndex].get( outputRow, outputColumn ) * filters[filterIndex].getWeight( depth, a, b );
					}
				}
			}
		}
		return error;
	}


	/**
	 * Compute the gradient of each weight in the 4D weight matrix (the filters)
	 */
	public void computeGradients() {
		for ( int k = 0 ; k < filters.length ; k++ ) {
			filters[k].computeGradient( input, delta[k], stride );
		}
	}


	@Override
	public void updateWeights( int batchSize ) {
		// todo. separate updateWeights and clearData
		for ( int i = 0 ; i < filters.length ; i++ ) {
			filters[i].updateWeights( batchSize );
		}
		super.clearData();
	}


	/*********************** GETTERS AND SETTERS ***************************************/

	public void setFilters( double[][][][] weights ) {
		filters = new Filter[weights.length];
		for ( int i = 0; i < weights.length ; i++ ) {
			Matrix[] matrixArr = new Matrix[weights[i].length];
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				matrixArr[j] = new Matrix( weights[i][j] );
			}
			filters[i] = new Filter( matrixArr );
		}
	}


//	public void printGradients() {
//		for ( int i = 0 ; i < filters.length ; i++ ) {
//			Utilities.print3DMatrix( filters[i].gradients );
//		}
//	}

}