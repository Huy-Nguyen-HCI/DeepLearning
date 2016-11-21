import Jama.Matrix;
import com.sun.xml.internal.ws.assembler.jaxws.MustUnderstandTubeFactory;

/**
 * Created by nguyenha on 11/6/2016.
 */
public class Filter {

    Matrix[] weights;
    Matrix[] gradients;
    int filterSize;
    double bias = 1;

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
//        gradients = Utilities.createMatrixWithSameDimension( weights );
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

    public void computeGradient( Matrix[] input, Matrix[] delta, int stride ) {
        for ( int depth = 0 ; depth < weights.length ; depth ++ ) {
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
