import java.util.ArrayList;
import java.util.Arrays;

public class Main {

	public static void main( String[] args ) {
		FullNeuralNetwork xorOperatorNeuralNetwork = new FullNeuralNetwork( 
			new int[]{2,2,2,1}, 
			new int[]{Neuron.STEP, Neuron.STEP, Neuron.STEP}, 
			new double[]{1, 1, 1} 
		);
		xorOperatorNeuralNetwork.setInputs( new double[]{1, 0} );
		xorOperatorNeuralNetwork.setWeights(
			new double[][][]{
				new double[][]{
					new double[]{1, 1, -0.5},
					new double[]{1, 1, -1,5}
				},
				new double[][]{
					new double[]{1, 0, 0},
					new double[]{0, -1, 0.5}
				},
				new double[][]{
					new double[]{1, 1, -1,5}
				}
			}
		);
		// printArray( xorOperatorNeuralNetwork.getOutputs() );
		SimulatedAnnealing s = new SimulatedAnnealing( xorOperatorNeuralNetwork );
		s.train();
	}
	
	public static void printArray(double[] arr) {
		for ( double x : arr ) System.out.println(x);
	}
}