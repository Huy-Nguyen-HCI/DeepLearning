import java.util.ArrayList;

public class ActivationFunctions {

	// types of activation function
	public static final int
		LINEAR = 0,
		STEP = 1,
		SIGMOID = 2,
		HYPERBOLIC = 3,
		RELU = 4,
		SOFTMAX = 5;

	/**
	 * Linear activation function used for regression neural networks.
	 * @param x the input value from the program or previous layer.
	 * @return the value passed by the neuron inputs.
	 */
	public static double linearAF( double x ) {
		return x;
	}


	public static double d_linearAF( double x ) {
		return 1;
	}


	/**
	 * Step activation function.
	 * @param x the input value from the program or previous layer.
	 * @return 1 for incoming values of 0.5 or higher and 0 otherwise.
	 */
	public static double stepAF( double x ) {
		return (x >= 0.5) ? 1 : 0;
	}


	/**
	 * Sigmoid activation function used for feedforward neural networks that ouput only positive numbers.
	 * @param x the input value from the program or previous layer.
	 * @return the sigmoid function.
	 */
	public static double sigmoidAF( double x ) {
		return 1.0 / (1 + Math.exp(-x));
	}


	public static double d_sigmoidAF( double x ) {
		double sigmoid = sigmoidAF( x );
		return sigmoid * (1 - sigmoid);
	}


	/**
	 * Hyperbolic activation function used for neural network that output values between -1 and 1.
	 * @param x the input value from the program or previous layer.
	 * @return the hyperbolic tangent function.
	 */
	public static double hyperbolicAF( double x ) {
		return Math.tanh(x);
	}


	public static double d_hyperbolicAF( double x ) {
		double hyperbolic = hyperbolicAF(x);
		return 1 - hyperbolic * hyperbolic;
	}


	/**
	 * Rectified linear unit, recommended by most current researches on hidden layers.
	 * @param x the input value from the program or previous layer.
	 * @return 0 if <tt>x</tt> is negative and <tt>x</tt> itself otherwise.
	 */
	public static double reLUAF( double x ) {
		return Math.max(0, x);
	}


	public static double d_reLUAF( double x ) {
		return (x > 0) ? 1 : 0;
	}


	/**
	 * Softmax activation function found in the output layer of a classification neural network.
	 * @param outputNeuron the array of output values from the previous layer.
	 * @return an array of probability that the input falls into each class.
	 */
	public static double[] softmaxAF( double[] outputNeuron ) {
		double sum = 0;
		for ( double v : outputNeuron) {
			sum += Math.exp(v);
		}
		double[] prob = new double[outputNeuron.length];
		for (int i = 0 ; i < prob.length ; i++) {
			prob[i] = Math.exp( outputNeuron[i] ) / sum;
		}
		return prob;
	}


	public static double[] d_softmaxAF( double[] outputNeuron ) {
		double[] prob = softmaxAF( outputNeuron );
		double[] derivative = new double[prob.length];
		for ( int i = 0 ; i < derivative.length ; i++ ) {
			derivative[i] = prob[i] * ( 1 - prob[i] );
		}
		return derivative;
	}


	public static double applyActivationFunction( int activationFunctionType, double input ) {		
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.linearAF( input );
			case STEP:
				return ActivationFunctions.stepAF( input );
			case SIGMOID:
				return ActivationFunctions.sigmoidAF( input );
			case HYPERBOLIC:
				return ActivationFunctions.hyperbolicAF( input );
			case RELU:
				return ActivationFunctions.reLUAF( input );
			default:
				assert false : "Error. Unrecognized activation function.";
				return -1;
		}
	}


	public static double applyActivationFunctionDerivative( int activationFunctionType, double input ) {
		switch ( activationFunctionType ) {
			case LINEAR:
				return ActivationFunctions.d_linearAF( input );
			case SIGMOID:
				return ActivationFunctions.d_sigmoidAF( input );
			case HYPERBOLIC:
				return ActivationFunctions.d_hyperbolicAF( input );
			case RELU:
				return ActivationFunctions.d_reLUAF( input );
			default:
				assert false : "Error. Unrecognized activation function.";
				return -1;
		}
	}

}