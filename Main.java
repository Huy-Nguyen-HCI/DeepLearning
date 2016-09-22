import java.util.Arrays;
import java.util.Random;

public class Main {

	public static void main( String[] args ) {
		FullNeuralNetwork andOperatorNeuralNetwork = new FullNeuralNetwork( 
			new int[]{2,2,1}, 
			new int[]{Neuron.SIGMOID, Neuron.SIGMOID}, 
			new double[]{ 1, 1} 
		);
		
				System.out.println( "\nFor this weight vector, the neural network outputs: " );
		andOperatorNeuralNetwork.setInputs( new double[]{1 ,1} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
		andOperatorNeuralNetwork.setInputs( new double[]{0 ,1} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
		andOperatorNeuralNetwork.setInputs( new double[]{1 ,0} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
		andOperatorNeuralNetwork.setInputs( new double[]{0 ,0} );
		printArray( andOperatorNeuralNetwork.getOutputs() );
		
		// training for AND
		SimulatedAnnealing s = new SimulatedAnnealing( 
			andOperatorNeuralNetwork,
			new double[][]{
				new double[]{1, 1},
				new double[]{1, 0},
				new double[]{0, 1},
				new double[]{0, 0}
			},
			new double[]{0, 1, 1, 0}
		);
		for (int i=0;i<30;i++)
		{
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
	}

	
	public static void printArray(double[] arr) {
		for ( double x : arr ) System.out.println(x);
	}


	public static double getRandomNumberInRange( double start, double end ) {
		double random = new Random().nextDouble();
		return start + (random * (end - start));
	}
}