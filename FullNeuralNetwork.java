import java.util.ArrayList;

public class FullNeuralNetwork {
	ArrayList<ArrayList<Neuron>> network;
	ArrayList<ArrayList<ArrayList<Double>>> weights;

	public FullNeuralNetwork( int[] numberOfNodesOnLayers, int[] activationFunctionTypes ) {
		// numberOfNodes.length layers, each of which is an array of Neuron
		network = new ArrayList<ArrayList<Neuron>>( numberOfNodesOnLayers.length );

		for ( int i = 0 ; i < numberOfNodesOnLayers.length ; i++ ) {
			// layer i has numberOfNodes[i] neurons
			network.set( i, new ArrayList<Neuron>(numberOfNodesOnLayers[i]) );
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
	

	public ArrayList<Double> getOutputsAtLayer( int layer ) {
		ArrayList<Double> outputs = new ArrayList<Double>();
		// base case
		if (layer == 0 ) {
			// get the outputs from the first layer, which are also the inputs
			for ( int i = 0 ; i < network.get(0).size(); i++ ) {
				outputs.add( network.get(0).output() );
			}
		}
		else {
			ArrayList<Double> prevLayerOutputs = getOutputsAtLayer( layer - 1 );
			for ( int i = 0 ; i < network.get(layer).size(); i++ ) {
				network.get(layer).
			}
		}
		return outputs;
	}
}