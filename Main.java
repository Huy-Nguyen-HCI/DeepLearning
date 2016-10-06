import java.util.Arrays;
import java.util.Random;

public class Main {

	public static void main( String[] args ) {
		FullNeuralNetwork example =  new FullNeuralNetwork(
			new int[]{2,2,1},
			new int[]{Neuron.SIGMOID, Neuron.SIGMOID},
			new double[]{1,1}
		);
		// example.setWeights(
		// 	new double[][][]{
		// 		new double[][]{
		// 			new double[]{0.15, 0.20, 0.35},
		// 			new double[]{0.25, 0.30, 0.35}
		// 		},
		// 		new double[][]{
		// 			new double[]{0.40, 0.45, 0.60},
		// 			new double[]{0.50, 0.55, 0.60}
		// 		}
		// 	}
		// );
		// double[][] inputs = new double[][]{new double[]{0.05, 0.10}};
		// double[][] targets = new double[][]{new double[]{0.01, 0.99}};
		// Backpropagation batch = new Backpropagation( example, inputs, targets );
		// batch.onlineTraining();
		// System.out.println( "output layer weights" );
		// Utilities.printArray( example.network[2][0].getWeights() );
		// Utilities.printArray( example.network[2][1].getWeights() );
		double[][] inputs = new double[][]{ 
			new double[]{1,1},
			new double[]{1,0},
			new double[]{0,1},
			new double[]{0,0} 
		};
		double[][] targets = new double[][]{
			new double[]{0},
			new double[]{1},
			new double[]{1},
			new double[]{0}
		};
		Backpropagation batch = new Backpropagation( example, inputs, targets );
		batch.batchTraining();
	}

}