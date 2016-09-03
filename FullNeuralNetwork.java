import java.util.ArrayList;

public class FullNeuralNetwork {
	ArrayList<ArrayList<Neuron>> network;
	ArrayList<ArrayList<ArrayList<Double>>> weights;

	public FullNeuralNetwork( int[] numberOfNodes, int[] activationFunctionTypes ) {
		// numberOfNodes.length layers, each of which is an array of Neuron
		network = new ArrayList<ArrayList<Neuron>>( numberOfNodes.length );

		for ( int i = 0 ; i < numberOfNodes.length ; i++ ) {
			// layer i has numberOfNodes[i] neurons
			network.set( i, new ArrayList<Neuron>(numberOfNodes[i]) );
			// initialize all the neurons in the layer
			for (int j = 0 ; j < network.get(i).size() ; j ++ ) {
				network.get(i).set( j, new Neuron() );
				network.get(i).get(j).setAFType( activationFunctionTypes[i] );
			}
		}
	}


	public void getInputs( double[] inputs ) {
		for ( int i = 0 ; i < network.get(0).size(); i++ ) {
			// first layer has input neurons
			network.get(0).get(i).setInput( inputs[i] );
		}
	}


	public void setWeights( ArrayList<ArrayList<ArrayList<Double>>> weights ) {
		this.weights = weights;
	}
	

	public void getOutput() {
		
	}
}