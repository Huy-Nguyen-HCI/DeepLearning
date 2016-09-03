public class Neuron {

	double input;
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


	public void setInput( double input ) {
		this.input = input;
	}


	public double output() {
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.linearAF( input );
			case STEP:
				return (int) ActivationFunctions.stepAF( input );
			case SIGMOID:
				return ActivationFunctions.sigmoidAF( input );
			case HYPERBOLIC:
				return ActivationFunctions.hyperbolicAF( input );
			case RELU:
				return ActivationFunctions.reLUAF( input );
			default:
				return input;
		}
	}
}