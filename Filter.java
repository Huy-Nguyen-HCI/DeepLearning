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
        gradients = new Matrix[FILTER_DEPTH];
        for ( int i = 0 ; i < FILTER_DEPTH ; i++ ) {
            weights[i] = new Matrix(filterSize, filterSize);
            gradients[i] = new Matrix(filterSize, filterSize);
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


    public void computeGradient( Matrix[] input, Matrix[] delta, int stride ) {
        for ( int depth = 0 ; depth < FILTER_DEPTH ; depth ++ ) {
            Matrix filterSlice = weights[depth];
            Matrix inputSlice = input[depth];
            Matrix deltaSlice = delta[depth];
            Matrix gradientSlice = gradients[depth];
            computeGradientAtSlice( filterSlice, inputSlice, deltaSlice, gradientSlice, stride);
        }
    }


    public void computeGradientAtSlice(
            Matrix filterSlice, Matrix inputSlice,
            Matrix deltaSlice,
            Matrix gradientSlice, int stride )
    {
        for ( int a = 0 ; a < filterSlice.getRowDimension() ; a++ ) {
            for ( int b = 0 ; b < filterSlice.getColumnDimension() ; b++ ) {
                // calculating gradient at weight (a,b)
                // loop through all neurons that are connected to this weight
                for ( int i = a ; a + i*stride < inputSlice.getRowDimension() ; i++ ) {
                    for ( int j = b ; b + j*stride < inputSlice.getColumnDimension() ; j++ ) {
                        double addition = deltaSlice.get(i,j) * inputSlice.get(i+a,j+b);
                        gradientSlice.set(i, j, gradientSlice.get(i,j) + addition);
                    }
                }
            }
        }
    }
}
