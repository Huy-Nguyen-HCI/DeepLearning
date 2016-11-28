import Jama.Matrix;

/**
 * Created by nguyenha on 11/6/2016.
 */
public class Filter {

    Matrix[] weights;
    Matrix[] gradients;
    int filterSize;
    double bias = 1;
    double biasGradient = 0;

    public Filter( int filterSize, int filterDepth ) {
        this.filterSize = filterSize;
        weights = new Matrix[filterDepth];
        for ( int i = 0 ; i < filterDepth ; i++ ) {
            weights[i] = new Matrix(filterSize, filterSize);
            for ( int row = 0 ; row < filterSize ; row++ ) {
                for ( int column = 0 ; column < filterSize ; column++ ) {
                    weights[i].set( row, column, Utilities.getRandomNumberInRange(-1,1) );
                }
            }
        }
        gradients = Utilities.createMatrixWithSameDimension( weights );
    }


    public Filter( Matrix[] weights ) {
        this.weights = weights;
        filterSize = weights[0].getRowDimension();
        gradients = Utilities.createMatrixWithSameDimension( weights );
    }

    /*********************** FEEDFORWARD ***************************************/

    public Matrix computeLinearCombination( Matrix[] input, int stride ) {
        int input2DSize = input[0].getRowDimension();
        int numberOfSteps = (input2DSize - filterSize) / stride + 1;
        Matrix output2DBoard = new Matrix(numberOfSteps, numberOfSteps);
        for ( int i = 0 ; i < numberOfSteps ; i++ ) {
            for ( int j = 0 ; j < numberOfSteps ; j++ ) {
                double linearCombinationSum = 0;
                // get one 2D slice of the input
                for ( int k = 0 ; k < weights.length  ; k++ ) {
                    Matrix mappedRegion = input[k].getMatrix(
                            stride * i,
                            stride * i + filterSize - 1,
                            stride * j,
                            stride * j + filterSize - 1
                    );
                    Matrix filter = weights[k];
                    // weights[k] gets mapped to a region in input[k]
                    linearCombinationSum += mappedRegion.arrayTimes(filter).sum();
                }
                output2DBoard.set( i, j, linearCombinationSum + bias );
            }
        }
        return output2DBoard;
    }


    /*********************** BACKPROPAGATION ***************************************/

    public void computeGradient( Matrix[] input, Matrix deltaSlice, int stride ) {
        for ( int depth = 0 ; depth <  weights.length; depth ++ ) {
            Matrix filterSlice = weights[depth];
            Matrix inputSlice = input[depth];
            Matrix gradientSlice = gradients[depth];

            // loop through all the weights in this filter slice
            for ( int a = 0 ; a < filterSize ; a++ ) {
                for ( int b = 0 ; b < filterSize ; b++ ) {
                    double gradient = 0;
                    int output2DSize = (inputSlice.getRowDimension() - filterSize) / stride + 1;
                    for ( int i = 0 ; i < output2DSize ; i++ ) {
                        for ( int j = 0 ; j < output2DSize ; j++ ) {
                            // weights at (a,b) connects to input neurons at (a + i*stride, b + j*stride)
                            // to output neurons at (i, j)
                            gradient += inputSlice.get( a + i*stride, b + j*stride ) * deltaSlice.get(i, j);
                        }
                    }
                    gradientSlice.set( a, b, gradient );
                }
            }
        }

        biasGradient = bias * deltaSlice.sum();
    }


    public void updateWeights() {
        for ( int k = 0 ; k < weights.length ; k++ ) {
            for ( int i = 0 ; i < weights[k].getRowDimension() ; i++ ) {
                for ( int j = 0 ; j < weights[k].getColumnDimension() ; j++ ) {
                    weights[k].set( i, j, weights[k].get(i,j) + gradients[k].get(i,j) );
                }
            }
        }
        clearData();
    }


    public void clearData() {
        // reinitialize the gradients matrix
        gradients = Utilities.createMatrixWithSameDimension( gradients );
        biasGradient = 0;
    }


    /*********************** GETTERS AND SETTERS **************/
    public double getWeight( int depth, int row, int column ) {
        return weights[depth].get( row, column );
    }
}
