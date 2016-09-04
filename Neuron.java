import java.util.ArrayList;

public class Neuron {

	ArrayList<Double> inputs;
	ArrayList<Double> weights;
	int activationFunctionType = -1;

	public static final int
		LINEAR = 0,
		STEP = 1,
		SIGMOID = 2,
		HYPERBOLIC = 3,
		RELU = 4,
		SOFTMAX = 5;

	// public Neuron( double input, int activationFunctionType ) {
	// 	this.input = input;
	// 	this.activationFunctionType = activationFunctionType;
	// }

	public void setAFType( int type ) {
		this.activationFunctionType = type;
	}


	public void setInput( ArrayList<Double> inputs) {
		this.inputs = inputs;
	}

	public void setWeights( ArrayList<Double> weights ) {
		this.weights = weights;
	}

	public double evaluateLinearCombination() {
		double linearCombination = 0;
		for (int i = 0 ; i < inputs.size(); i++ ) {
			linearCombination += inputs.get(i) * weights.get(i);
		}
		return linearCombination;
	}


	public double output() {
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.linearAF( evaluateLinearCombination() );
			case STEP:
				return (int) ActivationFunctions.stepAF( evaluateLinearCombination() );
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

	public int getAFType() {
		return activationFunctionType;
	}
}