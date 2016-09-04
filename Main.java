import java.util.ArrayList;
import java.util.Arrays;

public class Main {
	public static void main(String[] args) {
		FullNeuralNetwork andOperatorNeuralNetwork = 
			new FullNeuralNetwork( new int[]{2, 1}, new int[]{Neuron.STEP}, new double[]{1} );
		double[] w = new double[]{1, 1, -1.5};
		ArrayList<Double> weights = new ArrayList<Double>();
		for ( double x : w ) weights.add(x);
		andOperatorNeuralNetwork.network.get(1).get(0).setWeights( weights );
		andOperatorNeuralNetwork.setInputs( new double[]{1, 1} );
		ArrayList<Double> outputs = andOperatorNeuralNetwork.getOutputs();
		for ( double x : outputs ) System.out.println( x );
	}
}