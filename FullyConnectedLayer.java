import Jama.Matrix;
/**
 * Class that represents a fully connected layer in a feedforward or convolution NN.
 */
public class FullyConnectedLayer extends Layer {

	int neuronCount;
	double[][] weights;
	double[] oneDimensionalInput;

	double[] linearCombinations;
	double[] delta;


	public FullyConnectedLayer( int neuronCount, int activationFunction ) {
		super( activationFunction );
		this.neuronCount = neuronCount;
		delta = new double[neuronCount];
		linearCombinations = new double[neuronCount];
	}

	/************** FEEDFORWARD *****************/
	public void initializeWeights() {
		assert ( weights != null );
		for ( int i = 0 ; i < weights.length ; i++ ) {
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				weights[i][j] = Utilities.getRandomNumberInRange( -1, 1 );
			}
		}
	}

	public void setInput( Matrix[] input ) {
		this.input = input;
		int totalInputNeuron = input.length * input[0].getRowDimension() * input[0].getColumnDimension();
		weights = new double[neuronCount][totalInputNeuron];
		initializeWeights();
	}


	public void setInput( double[] input ) {
		oneDimensionalInput = input;
		weights = new double[neuronCount][input.length];
		initializeWeights();
	}


	public void getLinearCombination() {

		// if input is a 3D matrix
		if ( input != null ) {
			for ( int i = 0 ; i < output.length ; i++ ) {
				linearCombinations[i] = getLinearCombinationAtNeuron(i);
			}
		}

		assert( oneDimensionalInput != null );
		for ( int i = 0 ; i < output.length ; i++ ) {
			for ( int j = 0 ; j < oneDimensionalInput.length ; j++ ) {
				linearCombinations[i] += oneDimensionalInput[i] * weights[i][j];
			}
		}
	}


	public double getLinearCombinationAtNeuron( int neuronIndex ) {
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


	/************** BACKPROPAGATION *****************/
	public void calculateDeltaForOutputLayer( double[] target ) {
		for ( int i = 0 ; i < delta.length ; i++ ) {
			delta[i] =
					(target[i] - linearCombinations[i]) *
					ActivationFunctions.applyActivationFunctionDerivative( activationFunction, linearCombinations[i] );
		}
	}


	public void calculateDeltaForHiddenLayer( double[] nextLayerDelta, double[][] nextLayerWeights ) {
		for ( int i = 0 ; i < delta.length ; i++ ) {
			double derivative =
					ActivationFunctions.applyActivationFunctionDerivative( activationFunction, linearCombinations[i] );
			double deltaSum = 0;
			for ( int j = 0 ; j < nextLayerDelta.length ; j++ ) {
				deltaSum += nextLayerWeights[j][i] * nextLayerDelta[j];
			}
			delta[i] = deltaSum;
		}
	}
}