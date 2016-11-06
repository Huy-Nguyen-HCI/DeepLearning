import java.util.Arrays;
import java.util.Random;

public class Main {

	public static void main( String[] args ) {
		FullNeuralNetwork example =  new FullNeuralNetwork(
			new int[]{2,2,1},
			new int[]{ActivationFunctions.SIGMOID, ActivationFunctions.SIGMOID},
			new double[]{1,1}
		);
		double[][] inputs = new double[][]{ 
			new double[]{1,1},
			new double[]{1,0},
			new double[]{0,1},
			new double[]{0,0} 
		};
		double[][] targets = new double[][]{
			new double[]{1},
			new double[]{0},
			new double[]{0},
			new double[]{0}
		};
		Backpropagation batch = new Backpropagation( example, inputs, targets );
		batch.onlineTraining();
	}

}