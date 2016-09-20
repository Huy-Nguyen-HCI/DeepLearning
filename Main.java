import java.util.ArrayList;
import java.util.Arrays;

public class Main {

	public static void main( String[] args ) {
		FullNeuralNetwork andOperatorNeuralNetwork = new FullNeuralNetwork( 
			new int[]{2, 1}, 
			new int[]{Neuron.SIGMOID}, 
			new double[]{1} 
		);
		// training for AND
		SimulatedAnnealing s = new SimulatedAnnealing( 
			andOperatorNeuralNetwork,
			new double[][]{
				new double[]{1, 1},
				new double[]{1, 0},
				new double[]{0, 1},
				new double[]{0, 0}
			},
			new double[]{1, 0, 0, 0}
		);
		s.train();
		System.out.println( "\nFor this weight vector, the neural network outputs: " );
		andOperatorNeuralNetwork.setInputs( new double[]{1 ,1} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
		andOperatorNeuralNetwork.setInputs( new double[]{0 ,1} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
		andOperatorNeuralNetwork.setInputs( new double[]{1 ,0} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
		andOperatorNeuralNetwork.setInputs( new double[]{0 ,0} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
	}
	
	public static void printArray(double[] arr) {
		for ( double x : arr ) System.out.println(x);
	}
}