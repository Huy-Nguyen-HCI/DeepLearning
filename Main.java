import java.util.ArrayList;
import java.util.Arrays;

public class Main {
	public static void main(String[] args) {
		FullNeuralNetwork andOperatorNeuralNetwork = 
			new FullNeuralNetwork( new int[]{2,1}, new int[]{Neuron.STEP}, new double[]{1} );
		andOperatorNeuralNetwork.setWeightsForNeuron( 1, 0, new double[]{1, 1, -1.5} );
		andOperatorNeuralNetwork.setInputs( new double[]{1, 0} );
		double[] outputs = andOperatorNeuralNetwork.getOutputs();
		printArray(outputs);
	}

	public static void printArray(double[] arr) {
		for ( double x : arr ) System.out.println(x);
	}
}