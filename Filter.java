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
        int input2DHeight = input[0].getColumnDimension();
        int numberOfSteps = (input2DWidth - filterSize) / stride + 1;
        Matrix output2DBoard = new Matrix(numberOfSteps, numberOfSteps);
        for ( int k = 0 ; k < FILTER_DEPTH ; k++ ) {
            double linearCombinationSum = 0;
            for ( int i = 0 ; i < numberOfSteps ; i++ ) {
                for ( int j = 0 ; j < numberOfSteps ; j++ ) {
                    Matrix mappedRegion = input[k].getMatrix(
                            stride * i,
                            stride * i + filterSize,
                            stride * j,
                            stride * j + filterSize
                    );
                    linearCombinationSum += mappedRegion.arrayTimes(weights[k]).sum();
                }
            }
            output2DBoard.set( i, j, linearCombinationSum );
        }
    }

}
