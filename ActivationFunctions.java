import java.util.ArrayList;

public class ActivationFunctions {

	/**
	 * Linear activation function used for regression neural networks.
	 * @param x the input value from the program or previous layer.
	 * @return the value passed by the neuron inputs.
	 */
	public static double linearAF( double x ) {
		return x;
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


	/**
	 * Hyperbolic activation function used for neural network that output values between -1 and 1.
	 * @param x the input value from the program or previous layer.
	 * @return the hyperbolic tangent function.
	 */
	public static double hyperbolicAF( double x ) {
		return Math.tanh(x);
	}


	/**
	 * Rectified linear unit, recommended by most current researches on hidden layers.
	 * @param x the input value from the program or previous layer.
	 * @return 0 if <tt>x</tt> is negative and <tt>x</tt> itself otherwise.
	 */
	public static double reLUAF( double x ) {
		return Math.max(0, x);
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


	/**
	 * Sigmoid function with bias in a single-input neural network.
	 * @param x the input value.
	 * @param w the weight of <tt>x</tt>.
	 * @param bias a constant to adjust the slope or shape of the activation function.
	 */
	public static double singleInputNeuralNetwork( double x, double weight, double bias ) {
		return sigmoidAF( weight * x + bias );
	}

}