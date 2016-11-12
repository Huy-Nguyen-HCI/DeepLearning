import Jama.Matrix;
/**
 * Class that represents a fully connected layer in a feedforward or convolution NN.
 */
public class FullyConnectedLayer extends Layer {

	int neuronCount;
	double[][] weights;
	double[] oneDimensionalInput;

	public FullyConnectedLayer( int neuronCount, int activationFunction ) {
		super( activationFunction );
		this.neuronCount = neuronCount;
	}

	public void setInput( Matrix[] input ) {
		this.input = input;
		int totalInputNeuron = input.length * input[0].getRowDimension() * input[0].getColumnDimension();
		weights = new double[neuronCount][totalInputNeuron];
	}


	public void setInput( double[] input ) {
		oneDimensionalInput = input;
		weights = new double[neuronCount][input.length];
	}


	public double[] getOutput() {
		double[] output = new double[neuronCount];

		// if input is a 3D matrix
		if ( input != null ) {
			double linearCombination = 0;
			for ( int i = 0 ; i < output.length ; i++ ) {
				output[i] = getOutputAtNeuron(i);
			}
			return output;
		}

		assert( oneDimensionalInput != null );
		for ( int i = 0 ; i < output.length ; i++ ) {
			for ( int j = 0 ; j < oneDimensionalInput.length ; j++ ) {
				output[i] += oneDimensionalInput[i] * weights[i][j];
			}
		}
		return output;
	}


	public double getOutputAtNeuron( int neuronIndex ) {
		double output = 0;
		int count = 0;
		for ( int k = 0 ; k < input.length ; k++ ) {
			for ( int i = 0 ; i < input[k].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < input[k].getColumnDimension() ; j++ ) {
					output += weights[neuronIndex][count] * input[k].get( i, j );
					count ++;
				}
			}
		}
		return output;
	}
}