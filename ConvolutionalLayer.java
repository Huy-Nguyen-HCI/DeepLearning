import Jama.Matrix;

/**
 * Class that represents a convolutional layer in a convolutional neural network.
 */
public class ConvolutionalLayer extends Layer {

	Matrix[] filters;
	int stride;
	int padding;

	/**
	* Take a 3D matrix as input.
	*/
	public ConvolutionalLayer( Matrix[] filters, int stride, int activationFunction ) {
		super( activationFunction );
	}


	/**
	 * Return the padding p such that steps = (w - f + 2p) / (s+1) is an integer
	 * @param w width of the input image
	 * @return the value of p.
	 */
	private int getPadding( int w ) {
		return 0;
	}


	/**
	 * Get the number of steps needed for each filter to scan through one row
	 * @param w width of the input image
	 * @return (w - f + 2p) / (s+1)
	 */
	public int getNumberOfSteps( int w ) {
		int p = getPadding( w );
		int f = filters[0].getRowDimension();
		return ( w - f + 2*p ) / (stride + 1);
	}
}