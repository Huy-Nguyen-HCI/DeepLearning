public class LossFunction {

	public static double logLoss( double[] predicted, double[] actual ) {
		assert( predicted.length == actual.length );
		int n = predicted.length;
		double result = 0;
		for (int i = 0 ; i < n ; i++) {
			result += actual[i] * Math.log(predicted[i]) + (1 - actual[i]) * Math.log(1 - predicted[i]);
		}
		result = result * -1 / n;
		return result;
	}

	public static double multiClassLogLoss( double[][] predicted, double[][] actual) {
		assert( predicted.length == actual.length && predicted[0].length == actual[0].length );
		int n = predicted.length;
		int m = predicted[0].length;
		double result = 0;
		for (int i = 0 ; i < n ; i++) {
			for (int j = 0 ; j < m; j++) {
				result += actual[i][j] * Math.log(predicted[i][j]);
			}
		}
		result = result * -1 / n;
		return result;
	}

	public static double meanSquareError( double[] predicted, double[] actual ) {
		assert( predicted.length == actual.length );
		int n = predicted.length;
		double result = 0;
		for (int i = 0 ; i < n ; i++) {
			result += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
		}
		result = result / n;
		return result;
	}


	public static double crossEntropyError( double [] predicted, double[] actual ) {
		assert( predicted.length == actual.length );
		int n = predicted.length;
		double result = 0;
		for (int i = 0 ; i < n ; i++) {
			result += actual[i] * Math.log(predicted[i]) + (1 - actual[i]) * Math.log(1 - predicted[i]);
		}
		result = result * -1 / n;
		return result;
	}

}