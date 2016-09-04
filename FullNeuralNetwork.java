import java.util.ArrayList;

public class FullNeuralNetwork {
	ArrayList<ArrayList<Neuron>> network;
	ArrayList<ArrayList<ArrayList<Double>>> weights;

	/**
	 * Class constructor. Initializes the neural network based on the number of neuron and activation function for each layer.
	 * @param numberOfNodesOnLayers an array that contains the length of each layer.
	 * @param activationFunctionTypes an array that contains the activation function type for each layer.
	 */
	public FullNeuralNetwork( int[] numberOfNodesOnLayers, int[] activationFunctionTypes ) {
		// numberOfNodes.length layers, each of which is an array of Neuron
		network = new ArrayList<ArrayList<Neuron>>();

		for ( int i = 0 ; i < numberOfNodesOnLayers.length ; i++ ) {
			// layer i has numberOfNodes[i] neurons
			network.add( new ArrayList<Neuron>(numberOfNodesOnLayers[i]) );
			// initialize all the neurons in the layer
			if ( i == 0 ) {
				initInputNeuronLayer( numberOfNodesOnLayers[0] );
			}
			else {
				initOtherNeuronLayer( numberOfNodesOnLayers[i], i, activationFunctionTypes[i - 1] );
			}
		}
	}

	/**
	 * Class constructor. Initializes the neural network based on the number of neuron and activation function for each layer. 
	 * Also add a bias neuron with specified bias to each layer.
	 * @param numberOfNodesOnLayers an array that contains the length of each layer.
	 * @param activationFunctionTypes an array that contains the activation function type for each layer.
	 */
	public FullNeuralNetwork( int[] numberOfNodesOnLayers, int[] activationFunctionTypes, double[] bias ) {
		this( numberOfNodesOnLayers, activationFunctionTypes );
		for ( int i = 0 ; i < network.size() ; i++ ) {
			if ( i < bias.length ) {
				ArrayList<Neuron> layer = network.get(i);
				layer.add( new BiasNeuron(bias[i]) );
			}
		}
	}

	public void initOtherNeuronLayer( int layerSize, int layerIndex, int activationFunctionType ) {
		for (int j = 0 ; j < layerSize; j++) {
			network.get(layerIndex).add( new Neuron() );
			network.get(layerIndex).get(j).setAFType( activationFunctionType );
		}
	}

	public void initInputNeuronLayer( int layerSize ) {
		for (int j = 0 ; j < layerSize; j++ ) {
			network.get(0).add( new InputNeuron() );
		}
	}

	public void setInputs( double[] inputs ) {
		for ( int i = 0 ; i < network.get(0).size(); i++ ) {
			// first layer has input neurons
			Neuron n = network.get(0).get(i);
			if ( n instanceof InputNeuron ) {
				((InputNeuron) n).setInput( inputs[i] );
			}
		}
	}


	public void setWeights( ArrayList<ArrayList<ArrayList<Double>>> weights ) {
		for ( int i = 0 ; i < network.size() ; i++ ) {
			for (int j = 0 ; j < network.get(i).size(); i++ ) {
				network.get(i).get(j).setWeights( weights.get(i).get(j) );
			}
		}
	}
	

	public ArrayList<Double> getOutputsAtLayer( int layerIndex ) {
		ArrayList<Double> outputs = new ArrayList<Double>();
		ArrayList<Neuron> neuronLayer = network.get(layerIndex);
		// base case
		if (layerIndex == 0 ) {
			// get the outputs from the first layer, which are also the inputs
			for ( int i = 0 ; i < neuronLayer.size(); i++ ) {
				outputs.add( neuronLayer.get(i).output() );
			}
		}
		else {
			ArrayList<Double> prevLayerOutputs = getOutputsAtLayer( layerIndex - 1 );
			// softmax: need to compute output of all neurons at once
			if ( neuronLayer.get(0).getAFType() == Neuron.SOFTMAX ) {
				outputs = ActivationFunctions.softmaxAF( prevLayerOutputs );
			}
			// otherwise, compute output of each neuron independently and add to the output array
			else {
				for ( int i = 0 ; i < neuronLayer.size(); i++ ) {
					Neuron n = neuronLayer.get(i);
					if ( !(n instanceof BiasNeuron) ) {
						// give the neuron an array of inputs and array of weights
						n.setInput( prevLayerOutputs );
					}
					outputs.add( n.output() );
				}
			}
		}
		return outputs;
	}

	public ArrayList<Double> getOutputs() {
		return getOutputsAtLayer( network.size() - 1);
	}
}