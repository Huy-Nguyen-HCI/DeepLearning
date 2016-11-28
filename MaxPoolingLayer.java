import Jama.Matrix;

import javax.annotation.PostConstruct;
import java.util.HashMap;

/**
 * Class that represents a maxpooling layer in a convolutional network.
 */
public class MaxPoolingLayer extends Layer {

	int spatialExtent;
	int stride;
	Matrix[] error;
	HashMap<Position, Position> maxPositions = new HashMap<>();

	public MaxPoolingLayer( int spatialExtent, int stride ) {
		this.spatialExtent = spatialExtent;
		this.stride = stride;
	}

	/*********************** FEEDFORWARD ***************************************/

	/**
	 * Return a 3D matrix of smaller dimension after maxpooling
	 */
	public Matrix[] computeOutput() {
		Matrix[] output = new Matrix[input.length];
		for ( int k = 0 ; k < input.length ; k++ ) {
			int numberOfSteps = ( input[k].getRowDimension() - spatialExtent ) / stride + 1;
			output[k] = new Matrix( numberOfSteps , numberOfSteps );
			// downsample input[k] to output[k]
			for ( int i = 0 ; i < numberOfSteps ; i++ ) {
				for ( int j = 0 ; j < numberOfSteps ; j++ ) {
					// look at a 2D board at depth k
					Matrix mappedRegion = input[k].getMatrix(
							stride * i,
							stride * i + spatialExtent - 1,
							stride * j,
							stride * j + spatialExtent - 1
					);
					int[] maxPosition = mappedRegion.findPositionOfMax();
					double max = mappedRegion.findMax();
					output[k].set( i, j, max );
					// (i,j) takes value from ( maxPosition[0], maxPosition[1] )
					maxPositions.put(
							new Position(maxPosition[0],maxPosition[1], k),
							new Position( i, j, k )
					);
				}
			}
		}
		return output;
	}


	/*********************** BACKPROPAGATION ***************************************/
	public void setError( Matrix[] error ) {
		this.error = error;
	}


	@Override
	public Matrix[] propagateThreeDimensionalError() {
		Matrix[] propagatedError = Utilities.createMatrixWithSameDimension( input );
		for ( int k = 0 ; k < propagatedError.length ; k++ ) {
			for ( int i = 0 ; i < propagatedError[k].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < propagatedError[k].getColumnDimension() ; j++ ) {
					Position pos = new Position( k, i, j );
					if ( maxPositions.containsKey(pos) ) {
						Position errorSource = maxPositions.get(pos);
						propagatedError[k].set( i, j, error[errorSource.depth].get( errorSource.row, errorSource.column ) );
					}
					else {
						propagatedError[k].set( i, j, 0 );
					}
				}
			}
		}
		return propagatedError;
	}


	public void clearData() {
		super.clearData();
		error = null;
		maxPositions = new HashMap<>();
	}
}