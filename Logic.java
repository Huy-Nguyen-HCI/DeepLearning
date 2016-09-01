

public class Logic {

	/**
	 * The logical AND operator evaluated by step activation function.
	 * @param x1 the first boolean variable (1 for TRUE, 0 for FALSE).
	 * @param x2 the second boolean variable (1 for TRUE, 0 for FALSE).
	 * @param bias the bias to tweak the result as desired, default to 1.
	 * @return the result of x1 AND x2.
	 */
	static int and( int x1, int x2, int bias ) {
		return ActivationFunctions.stepAF( x1 * 1 + x2 * 1 + bias * (-1.5) );
	}

	/**
	 * The logical OR operator evaluated by step activation function.
	 * @param x1 the first boolean variable (1 for TRUE, 0 for FALSE).
	 * @param x2 the second boolean variable (1 for TRUE, 0 for FALSE).
	 * @param bias the bias to tweak the result as desired, default to 1.
	 * @return the result of x1 OR x2.
	 */
	static int or( int x1, int x2, int bias ) {
		return ActivationFunctions.stepAF( x1 * 1 + x2 * 1 + bias * (-1.5) );
	}

	/**
	 * The logical NOT operator evaluated by step activation function.
	 * @param x1 the input boolean variable (1 for TRUE, 0 for FALSE).
	 * @param bias the bias to tweak the result as desired, default to 1.
	 * @return the result of NOT x1.
	 */
	static int not( int x1, int bias ) {
		return ActivationFunctions.stepAF( x1 * (-1) + bias * 0.5 );
	}

	/**
	 * The logical XOR operator evaluated by step activation function.
	 * @param x1 the first boolean variable (1 for TRUE, 0 for FALSE).
	 * @param x2 the second boolean variable (1 for TRUE, 0 for FALSE).
	 * @param bias the bias to tweak the result as desired, default to 1.
	 * @return the result of x1 XOR x2.
	 */
	static int xor( int x1, int x2, int bias ) {
		int y1 = ActivationFunctions.stepAF( x1 * 1 + x2 * 1 + bias * (-0.5) );
		int y2 = ActivationFunctions.stepAF( x1 * 1 + x2 * 1 + bias * (-1.5) );
		int z1 = ActivationFunctions.stepAF( y1 * (-1) + bias * 0.5 );
		return ActivationFunctions.stepAF( y1 * 1 + z1 * 1 + bias * (-1.5) );
	}
}