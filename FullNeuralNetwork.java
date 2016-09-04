import java.util.ArrayList;

public class FullNeuralNetwork {
	Neuron[][] network;
	double[][][] weights;

	/**
	 * Class constructor. Initializes the neural network based on the number of neuron and activation function for each layer.
	 * @param numberOfNodesOnLayers an array that contains the length of each layer.
	 * @param activationFunctionTypes an array that contains the activation function type for each layer.
	 */
	public FullNeuralNetwork( int[] numberOfNodesOnLayers, int[] activationFunctionTypes ) {
		// each layer except for the input layer must have an activation function
		assert( activationFunctionTypes.length + 1 == numberOfNodesOnLayers.length );

		// numberOfNodes.length layers, each of which is an array of Neuron
		network = new Neuron[numberOfNodesOnLayers.length][];

		for ( int i = 0 ; i < numberOfNodesOnLayers.length ; i++ ) {
			// layer i has numberOfNodes[i] neurons
			network[i] = new Neuron[numberOfNodesOnLayers[i]];
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
		// each layer except for the input layer must have an activation function
		assert( activationFunctionTypes.length + 1 == numberOfNodesOnLayers.length );

		// numberOfNodesOnLayers.length layers, each of which is an array of Neuron
		network = new Neuron[numberOfNodesOnLayers.length][];
		// initialize neurons
		for ( int i = 0 ; i < numberOfNodesOnLayers.length ; i++ ) {
			// layer i has numberOfNodes[i] + 1 neurons, the last one is a bias
			int size = ( i < bias.length ) ? numberOfNodesOnLayers[i] + 1 : numberOfNodesOnLayers[i];
			network[i] = new Neuron[size];
			// initialize all the neurons in the layer
			if ( i == 0 ) {
				initInputNeuronLayer( numberOfNodesOnLayers[0] );
			}
			else {
				initOtherNeuronLayer( numberOfNodesOnLayers[i], i, activationFunctionTypes[i - 1] );
			}
		}
		// add bias neurons
		for ( int i = 0 ; i < bias.length; i++ ) {
			Neuron[] layer = network[i];
			layer[layer.length - 1] = new BiasNeuron(bias[i]);
		}
	}


	public void initInputNeuronLayer( int layerSize ) {
		for (int j = 0 ; j < layerSize; j++ ) {
			network[0][j] = new InputNeuron();
		}
	}


	public void initOtherNeuronLayer( int layerSize, int layerIndex, int activationFunctionType ) {
		for (int j = 0 ; j < layerSize; j++) {
			network[layerIndex][j] = new Neuron();
			network[layerIndex][j].setAFType( activationFunctionType );
		}
	}


	public void setInputs( double[] inputs ) {
		for ( int i = 0 ; i < network[0].length; i++ ) {
			// first layer has input neurons
			Neuron n = network[0][i];
			if ( n instanceof InputNeuron ) {
				((InputNeuron) n).setInput( inputs[i] );
			}
		}
	}


	public void setWeights( double[][][] weights ) {
		// every neuron except input neuron must have an array of weights
		assert( weights.length == network.length - 1 );
		for ( int i = 0 ; i < network.length ; i++ ) {
			for (int j = 0 ; j < network[i].length; i++ ) {
				setWeightsForNeuron(i, j, weights[i][j] );
			}
		}
	}


	public void setWeightsForNeuron( int layerIndex, int index, double[] weights ) {
		network[layerIndex][index].setWeights( weights );
	}
	

	public double[] getOutputsAtLayer( int layerIndex ) {
		Neuron[] neuronLayer = network[layerIndex];
		double[] outputs = new double[neuronLayer.length];
		// base case
		if (layerIndex == 0 ) {
			// get the outputs from the first layer, which are also the inputs
			for ( int i = 0 ; i < neuronLayer.length; i++ ) {
				outputs[i] = neuronLayer[i].output();
			}
		}
		else {
			double[] prevLayerOutputs = getOutputsAtLayer( layerIndex - 1 );
			// softmax: need to compute output of all neurons at once
			if ( neuronLayer[0].getAFType() == Neuron.SOFTMAX ) {
				outputs = ActivationFunctions.softmaxAF( prevLayerOutputs );
			}
			// otherwise, compute output of each neuron independently and add to the output array
			else {
				for ( int i = 0 ; i < neuronLayer.length; i++ ) {
					Neuron n = neuronLayer[i];
					if ( !(n instanceof BiasNeuron) ) {
						// give the neuron an array of inputs and array of weights
						n.setInput( prevLayerOutputs );
					}
					outputs[i] =  n.output();
				}
			}
		}
		return outputs;
	}


	public double[] getOutputs() {
		return getOutputsAtLayer( network.length - 1);
	}

}