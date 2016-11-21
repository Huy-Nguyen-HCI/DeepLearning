import Jama.Matrix;

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


	@Override
	public void setInput( Matrix[] input ) {
		this.input = input;
		padInput( padding );
		for ( int i = 0 ; i < filters.length ; i++ ) {
			filters[i] = new Filter( filterSize, input.length );
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


	/**
	 *
	 * @return
	 */
//	public Matrix[] propagateError() {
//        Matrix[] error = Utilities.createMatrixWithSameDimension( input );
//		for ( int inputDepth = 0 ; inputDepth < input.length ; inputDepth ++ ) {
//			for ( int i = padding ; i < input[inputDepth].getRowDimension() - padding ; i++ ) {
//				for ( int j = padding ; j < input[inputDepth].getColumnDimension() - padding ; j++ ) {
//					// take all the weights connected to input[k][i][j]
//					double err = 0;
//					for ( int a = 0 ; a < filterSize ; a++ ) {
//						for ( int b = 0 ; b < filterSize ; b++ ) {
//							int x = i - a*stride;
//							int y = j - b*stride;
//							if ( x >= 0 && y < output[inputDepth].getRowDimension() && y >= 0 &&
//									y < output[inputDepth].getColumnDimension()
//							)
//							{
//								for ( int filterN = 0 ; filterN < filters.length ; filterN++ ) {
//									err += delta[filterN].get(x,y) * filters[filterN].weights[inputDepth].get(x,y);
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//        return error;
//	}

//	public Matrix[] propagateError() {
//		// create a matrix with the same size as the original input
//		Matrix[] propagatedError = new Matrix[input.length];
//		for ( int i = 0 ; i < input.length ; i++ ) {
//			Matrix paddedInput = input[i];
//			propagatedError[i] = new Matrix( paddedInput.getRowDimension() - 2*padding, paddedInput.getColumnDimension() - 2*padding );
//		}
//		int input2DSize = input[0].getRowDimension();
//		int output2DSize = (input2DSize - filterSize) / stride + 1;
//
//		// calculate error for each neuron in the input
//		for ( int k = 0 ; k < propagatedError.length ; k++ ) {
//			for ( int i = 0 ; i < propagatedError[k].getRowDimension() ; i++ ) {
//				for ( int j = 0 ; j < propagatedError[k].getColumnDimension() ; j++ ) {
//					// take all the weights connected to input[k][i][j]
//					double err = 0;
//					for ( int a = 0 ; a < filterSize ; a++ ) {
//						for ( int b = 0 ; b < filterSize ; b++ ) {
//							int row = i - a * stride;
//							int column = j - b * stride;
//							if ( row >= 0 && row < output2DSize && column >= 0 && column < output2DSize ) {
//
//							}
//						}
//					}
//				}
//			}
//		}
//		return propagatedError;
//	}



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

}