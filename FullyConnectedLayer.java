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
	}

	/*********************** FEEDFORWARD ***********************/
	public void initializeWeights() {
		for ( int i = 0 ; i < weights.length ; i++ ) {
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				weights[i][j] = Utilities.getRandomNumberInRange( -1, 1 );
			}
			System.out.println("weights at " + i + " is: " );
			Utilities.printArray( weights[i] );
		}
		initializeGradients();
	}


	private void initializeGradients() {
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


	public void computeLinearCombinations() {
		// if input is a 3D matrix
		if ( input != null && oneDimensionalInput == null ) {
			for ( int i = 0 ; i < linearCombinations.length ; i++ ) {
				linearCombinations[i] = getLinearCombinationAtNeuron(i);
			}
		}
		else {
			assert( oneDimensionalInput != null );
			for ( int i = 0 ; i < linearCombinations.length ; i++ ) {
				for ( int j = 0 ; j < oneDimensionalInput.length ; j++ ) {
					linearCombinations[i] += oneDimensionalInput[j] * weights[i][j];
				}
			}
		}
		System.out.println( "linear combination is: " );
		Utilities.printArray( linearCombinations );
	}


	public double[] computeOutput() {
		if ( activationFunction == ActivationFunctions.SOFTMAX ) {
			return ActivationFunctions.softmaxAF( linearCombinations );
		}
		double[] outputs = new double[linearCombinations.length];
		for ( int i = 0 ; i < linearCombinations.length ; i++ ) {
			outputs[i] = ActivationFunctions.applyActivationFunction( activationFunction, linearCombinations[i] );
		}
		return outputs;
	}


	private double getLinearCombinationAtNeuron( int neuronIndex ) {
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


	/*********************** BACKPROPAGATION ***********************/

	public void setErrorAndComputeDeltas( double[] error ) {
		assert ( activationFunction != ActivationFunctions.SOFTMAX );
		for ( int i = 0 ; i < delta.length ; i++ ) {
			delta[i] = error[i] * ActivationFunctions.applyActivationFunctionDerivative( activationFunction, linearCombinations[i] );
		}
	}


	public void computeNodeDeltasForOutputLayer( double[] target ) {
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


	public void computeGradients() {
		for ( int i = 0 ; i < weights.length ; i++ ) {
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				double gradient = delta[i] * getInputBeforeFlattened( j );
				gradients[i][j] += gradient;
			}
		}
	}


	public void updateWeights() {
		for ( int i = 0 ; i < weights.length ; i++ ) {
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				weights[i][j] += gradients[i][j];
			}
		}
		clearData();
	}


	public void clearData() {
		super.clearData();
		oneDimensionalInput = null;
		linearCombinations = new double[linearCombinations.length];
	}


	private double getInputBeforeFlattened( int index ) {
		if ( oneDimensionalInput != null )
			return oneDimensionalInput[index];

		int oneDimensionalSize = input[0].getRowDimension();
		int twoDimensionalSize = oneDimensionalSize * oneDimensionalSize;

		int depth = index / twoDimensionalSize;
		index = index % twoDimensionalSize;
		int row = index / oneDimensionalSize;
		index = index % oneDimensionalSize;
		int column = index;
		return input[depth].get( row, column);
	}


	@Override
	public double[] propagateOneDimensionalError() {
		assert( oneDimensionalInput != null );
		double[] propagatedError = new double[oneDimensionalInput.length];
		for ( int i = 0 ; i < propagatedError.length ; i++ ) {
			for ( int j = 0 ; j < delta.length ; j++ ) {
				propagatedError[i] += weights[j][i] * delta[j];
			}
		}
		return propagatedError;
	}


	@Override
	public Matrix[] propagateThreeDimensionalError() {
		int oneDimensionalSize = input[0].getRowDimension();
		int twoDimensionalSize = oneDimensionalSize * oneDimensionalSize;

		Matrix[] error = Utilities.createMatrixWithSameDimension( input );
		for ( int k = 0 ; k < error.length ; k++ ) {
			for ( int i = 0 ; i < error[k].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < error[k].getColumnDimension() ; j++ ) {
					double err = 0;
					for ( int nodeIndex = 0 ; nodeIndex < delta.length ; nodeIndex ++ ) {
						err += delta[nodeIndex] * weights[nodeIndex][ k * twoDimensionalSize + i * oneDimensionalSize + j ];
					}
					error[k].set( i, j, err );
				}
			}
		}
		System.out.println("3d propapagated:" );
		Utilities.print3DMatrix( error );
		return error;
	}
}