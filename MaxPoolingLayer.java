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

	public MaxPoolingLayer( int spatialExtent, int stride, int activationFunction ) {
		super( activationFunction );
		this.spatialExtent = spatialExtent;
		this.stride = stride;
	}

	/**
	 * Return a 3D matrix of smaller dimension after maxpooling
	 */
	public Matrix[] getOutput() {
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
					maxPositions.put(
							new Position(maxPosition[0],maxPosition[1], k),
							new Position( i, j, k )
					);
				}
			}
		}
		return output;
	}


	public void setError( Matrix[] error ) {
		this.error = error;
	}


	public Matrix[] propagateError() {
		Matrix[] propagatedError = Utilities.createMatrixWithSameDimension( input );
		for ( int k = 0 ; k < propagatedError.length ; k++ ) {
			for ( int i = 0 ; i < propagatedError[k].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < propagatedError[k].getColumnDimension() ; j++ ) {
					Position pos = new Position(i,j,k);
					double err = maxPositions.containsKey(pos) ? maxPositions.get(pos) : 0;
					propagatedError[k].set( i, j, err );
				}
			}
		}
		return propagatedError;
	}


	class Position {
		Integer x, y, z;

		Position( Integer x, Integer y, Integer z ) {
			this.x = x;
			this.y = y;
			this.z = z;
		}


		@Override
		public boolean equals( Object o ) {
			if ( !(o instanceof Position) ) return false;
			Position pos = (Position) o;
			return x.equals(pos.x) && y.equals(pos.y) && z.equals(pos.z);
		}

		@Override
		public int hashCode() {
			return x.hashCode() + y.hashCode() + z.hashCode();
		}
	}
}