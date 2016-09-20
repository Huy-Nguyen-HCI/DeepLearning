

public class Neuron {

	double[] inputs;
	double[] weights;
	int activationFunctionType = -1;
	double delta;

	// types of activation function
	public static final int
		LINEAR = 0,
		STEP = 1,
		SIGMOID = 2,
		HYPERBOLIC = 3,
		RELU = 4,
		SOFTMAX = 5;


	public void setAFType( int type ) {
		this.activationFunctionType = type;
	}


	public void setDelta( double delta ) {
		this.delta = delta;
	}


	public double getDelta() {
		return delta;
	}


	public void setInput( double[] inputs) {
		this.inputs = inputs;
	}


	public void setWeights( double[] weights ) {
		this.weights = weights;
	}


	public double[] getWeights() {
		return weights;
	}


	public double evaluateLinearCombination() {
		double linearCombination = 0;
		for (int i = 0 ; i < inputs.length; i++ ) {
			linearCombination += inputs[i] * weights[i];
		}
		return linearCombination;
	}


	public double output() {
		// todo. save output to a variable
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.linearAF( evaluateLinearCombination() );
			case STEP:
				return ActivationFunctions.stepAF( evaluateLinearCombination() );
			case SIGMOID:
				return ActivationFunctions.sigmoidAF( evaluateLinearCombination() );
			case HYPERBOLIC:
				return ActivationFunctions.hyperbolicAF( evaluateLinearCombination() );
			case RELU:
				return ActivationFunctions.reLUAF( evaluateLinearCombination() );
			default:
				System.err.println("Error. Undefined activation function");
				return -1;
		}
	}


	public double getAFDerivative() {
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.d_linearAF( output() );
			case SIGMOID:
				return ActivationFunctions.d_sigmoidAF( output() );
			case HYPERBOLIC:
				return ActivationFunctions.d_hyperbolicAF( output() );
			case RELU:
				return ActivationFunctions.d_reLUAF( output() );
			default:
				System.err.println("Error. Undefined activation function");
				return -1;
		}
	}


	public int getAFType() {
		return activationFunctionType;
	}
}