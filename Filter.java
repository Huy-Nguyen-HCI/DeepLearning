import Jama.Matrix;

/**
 * Created by nguyenha on 11/6/2016.
 */
public class Filter {
    public static final int FILTER_DEPTH = 3;
    Matrix[] weights;
    Matrix[] gradients;
    int filterSize;

    public Filter( int filterSize ) {
        this.filterSize = filterSize;
        weights = new Matrix[FILTER_DEPTH];
        for ( int i = 0 ; i < FILTER_DEPTH ; i++ ) {
            weights[i] = new Matrix(filterSize, filterSize);
        }
    }


    public Matrix computeLinearCombination( Matrix[] input, int stride ) {
        assert input.length == FILTER_DEPTH : "Input must have size K x K x 3";
        int input2DWidth = input[0].getRowDimension();
        int numberOfSteps = (input2DWidth - filterSize) / stride + 1;
        Matrix output2DBoard = new Matrix(numberOfSteps, numberOfSteps);
        for ( int i = 0 ; i < numberOfSteps ; i++ ) {
            for ( int j = 0 ; j < numberOfSteps ; j++ ) {
                double linearCombinationSum = 0;
                for ( int k = 0 ; k < input.length ; k++ ) {
                    Matrix mappedRegion = input[k].getMatrix(
                            stride * i,
                            stride * i + filterSize,
                            stride * j,
                            stride * j + filterSize
                    );
                    linearCombinationSum += mappedRegion.arrayTimes(weights[k]).sum();
                }
                output2DBoard.set( i, j, linearCombinationSum );
            }
        }
    }


    public void computeGradient( Matrix[] input, Matrix[] error, Matrix[] linearCombinations ) {
        for ( int depth = 0 ; depth < FILTER_DEPTH ; depth ++ ) {
            Matrix filterSlice = weights[depth];
            Matrix imageSlice = input[depth];
            Matrix errorSlice = error[depth];
            Matrix linearSlice = linearCombinations[depth];

        }
    }


    public void computeGradientAtSlice( Matrix filterSlice, Matrix inputSlice, Matrix errorSlice, Matrix linearSlice ) {
        for ( int a = 0 ; a < filterSlice.getRowDimension() ; a++ ) {
            for ( int b = 0 ; b < filterSlice.getColumnDimension() ; b++ ) {
                // calculating gradient at weight (a,b)
                // loop through all neurons that are connected to this weight
                for ( int i = 0 ; a + i*stride < inputSlice.getRowDimension() ; i++ ) {
                    for ( int j = 0 ; b + j*stride < inputSlice.getColumnDimension() ; j++ ) {
                        gradients[k].get(i,j) += computeNodeDelta(k,i,j) * input[k].get(i+a,j+b);
                    }
                }
            }
        }
    }

}
