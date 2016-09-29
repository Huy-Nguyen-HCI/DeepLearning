import java.util.Arrays;

public class Neuron {

	double[] inputs;
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



	public double evaluateLinearCombination( double[] weights ) {
		double linearCombination = 0;
		for (int i = 0 ; i < inputs.length; i++ ) {
			linearCombination += inputs[i] * weights[i];
		}
		return linearCombination;
	}


	public double output( double[] weights ) {
		// todo. save output to a variable
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.linearAF( evaluateLinearCombination( weights ) );
			case STEP:
				return ActivationFunctions.stepAF( evaluateLinearCombination( weights ) );
			case SIGMOID:
				return ActivationFunctions.sigmoidAF( evaluateLinearCombination( weights ) );
			case HYPERBOLIC:
				return ActivationFunctions.hyperbolicAF( evaluateLinearCombination( weights ) );
			case RELU:
				return ActivationFunctions.reLUAF( evaluateLinearCombination( weights ) );
			default:
				System.err.println("Error. Undefined activation function");
				return -1;
		}
	}


	public double getAFDerivative( double[] weights ) {
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.d_linearAF( evaluateLinearCombination( weights ) );
			case SIGMOID:
				return ActivationFunctions.d_sigmoidAF( evaluateLinearCombination( weights ) );
			case HYPERBOLIC:
				return ActivationFunctions.d_hyperbolicAF( evaluateLinearCombination( weights ) );
			case RELU:
				return ActivationFunctions.d_reLUAF( evaluateLinearCombination( weights ) );
			default:
				System.err.println("Error. Undefined activation function");
				return -1;
		}
	}


	public int getAFType() {
		return activationFunctionType;
	}
}