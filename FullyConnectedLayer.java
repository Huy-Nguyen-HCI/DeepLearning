import Jama.Matrix;
/**
 * Class that represents a fully connected layer in a feedforward or convolution NN.
 */
public class FullyConnectedLayer extends Layer {

	double[][] weights;
	double[] oneDimensionalInput;

	double[] linearCombinations;
	double[] delta;
	double[][] gradients;


	public FullyConnectedLayer( int neuronCount, int activationFunction ) {
		super( activationFunction );
		delta = new double[neuronCount];
		linearCombinations = new double[neuronCount];
		initializeWeights();

	}

	/************** FEEDFORWARD *****************/
	public void initializeWeights() {
		for ( int i = 0 ; i < weights.length ; i++ ) {
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				weights[i][j] = Utilities.getRandomNumberInRange( -1, 1 );
			}
		}
	}


	public void initializeGradients() {
		gradients = new double[weights.length][weights[0].length];
	}


	public void setInput( Matrix[] input ) {
		this.input = input;
		int totalInputNeuronNumber = input.length * input[0].getRowDimension() * input[0].getColumnDimension();
		weights = new double[delta.length][totalInputNeuronNumber];
		initializeWeights();
	}


	public void setInput( double[] input ) {
		oneDimensionalInput = input;
		weights = new double[delta.length][input.length];
		initializeWeights();
	}


	public void getLinearCombination() {

		// if input is a 3D matrix
		if ( input != null && oneDimensionalInput == null ) {
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
		if ( activationFunction == ActivationFunctions.SOFTMAX ) {
			delta = ActivationFunctions.d_softmaxAF( linearCombinations );
		}
		else {
			for ( int i = 0 ; i < delta.length ; i++ ) {
				delta[i] = (target[i] - linearCombinations[i]) *
						ActivationFunctions.applyActivationFunctionDerivative( activationFunction, linearCombinations[i] );
			}
		}
	}


	public void calculateDeltaForHiddenLayer( double[] nextLayerDelta, double[][] nextLayerWeights ) {
		assert( nextLayerDelta.length == nextLayerWeights.length );
		for ( int i = 0 ; i < delta.length ; i++ ) {
			double derivative =
					ActivationFunctions.applyActivationFunctionDerivative( activationFunction, linearCombinations[i] );
			double propagatedError = 0;
			for ( int j = 0 ; j < nextLayerDelta.length ; j++ ) {
				propagatedError += nextLayerWeights[j][i] * nextLayerDelta[j];
			}
			delta[i] = propagatedError * derivative;
		}
	}


	public void calculateGradients() {
		for ( int i = 0 ; i < weights.length ; i++ ) {
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				double gradient = delta[i] * getInputBeforeFlattened( j );
				gradients[i][j] += gradient;
			}
		}
	}


	public double getInputBeforeFlattened( int index ) {
		if ( oneDimensionalInput != null ) return oneDimensionalInput[index];
		int oneDimensionalSize = input[0].getRowDimension();
		int twoDimensionalSize = oneDimensionalSize * oneDimensionalSize;

		int depth = index / twoDimensionalSize;
		index = index % twoDimensionalSize;
		int row = index / oneDimensionalSize;
		index = index % oneDimensionalSize;
		int column = index;
		return input[depth].get( row, column);
	}


	public double[] propagateError() {
		return delta;
	}


	public Matrix[] propagateError() {
		int oneDimensionalSize = input[0].getRowDimension();
		int twoDimensionalSize = oneDimensionalSize * oneDimensionalSize;

		Matrix[] error = Utilities.createMatrixWithSameDimension( input );
		for ( int k = 0 ; k < error.length ; k++ ) {
			for ( int i = 0 ; i < error[i].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < error[j].getColumnDimension() ; j++ ) {
					double err = 0;
					for ( int nodeIndex = 0 ; nodeIndex < delta.length ; nodeIndex ++ ) {
						err += delta[nodeIndex] * weights[nodeIndex][ k * twoDimensionalSize + oneDimensionalSize ];
					}
				}
			}
		}
		return error;
	}


}