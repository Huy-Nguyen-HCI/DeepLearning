import java.util.Arrays;

public class FullNeuralNetwork {
	// types of loss function
	public static final int 
		MSE = 0,
		CROSS_ENTROPY = 1;
	int lossFunctionType = MSE; // default to be cross entropy
	Neuron[][] network;
	double[][][] weights;
	// save the new weights to an array because all the weights
	// need to be updated at once
	double[][][] tempWeights;
	// weight change from the previous iteration, used for backpropagation
	double[][][] weightDeltas; 
	double learningRate = 0.5, momentum = 0;

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
				initInputNeuronLayer();
			}
			else {
				initOtherNeuronLayer( i, activationFunctionTypes[i - 1] );
			}
		}

		initializeWeights();	
	}

	/**
	 * Class constructor. Initializes the neural network based on the number of neuron and activation function for each layer. 
	 * Also add a bias neuron with specified bias to each layer.
	 * @param numberOfNodesOnLayers an array that contains the length of each layer.
	 * @param activationFunctionTypes an array that contains the activation function type for each layer.
	 * @param bias an array of bias values for each layer starting at layer 0.
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
				initInputNeuronLayer();
			}
			else {
				initOtherNeuronLayer( i, activationFunctionTypes[i - 1] );
			}
		}
		// add bias neurons
		for ( int i = 0 ; i < bias.length; i++ ) {
			Neuron[] layer = network[i];
			layer[layer.length - 1] = new BiasNeuron(bias[i]);
		}

		initializeWeights();
	}


	/**
	 * Initializes all input neurons on the first layer.
	 */
	private void initInputNeuronLayer() {
		for (int j = 0 ; j < network[0].length; j++ ) {
			network[0][j] = new InputNeuron();
		}
	}


	/**
	 * Initializes all neurons on a subsequent layer.
	 * @param layerIndex the index of the layer.
	 * @param activationFunctionType the type of the layer's activation function.
	 */
	private void initOtherNeuronLayer( int layerIndex, int activationFunctionType ) {
		for (int j = 0 ; j < network[layerIndex].length; j++) {
			network[layerIndex][j] = new Neuron();
			network[layerIndex][j].setAFType( activationFunctionType );
		}
	}


	/**
	 * Evaluate the output for each neuron in the specified layer.
	 * @param layerIndex the index of the layer.
	 * @return an array of outputs from all neurons.
	 */
	private double[] getOutputsAtLayer( int layerIndex ) {
		Neuron[] neuronLayer = network[layerIndex];
		double[] outputs = new double[neuronLayer.length];
		// base case
		if (layerIndex == 0 ) {
			// get the outputs from the first layer, which are also the inputs
			for ( int i = 0 ; i < neuronLayer.length; i++ ) {
				Neuron n = neuronLayer[i];
				if (n instanceof BiasNeuron ) {
					outputs[i] = ((BiasNeuron) neuronLayer[i]).output();
				}
				else {
					outputs[i] = ((InputNeuron) neuronLayer[i]).output();
				}
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
						// give the neuron an array of inputs
						n.setInput( prevLayerOutputs );
						outputs[i] = n.output( weights[layerIndex - 1][i] );
					}
					else {
						outputs[i] = ( (BiasNeuron) n).output();
					}
				}
			}
		}
		return outputs;
	}


	/**
	 * Get the output of the network.
	 * @return the array output of the final layer in the network.
	 */
	public double[] getOutputs() {
		return getOutputsAtLayer( network.length - 1 );
	}


	/**
	 * Initialize the weight matrix to contain all 0s.
	 */
	public void initializeWeights() {
		// nodes on all layers except the input layers have weights
		weights = new double[ network.length - 1][][];
		for ( int i = 1 ; i < network.length ; i++ ) {
			// each node has its array of weights
			weights[i - 1] = new double[network[i].length][];
			for ( int j = 0 ; j < weights[i - 1].length ; j++ ) {
				// each node is connected to all nodes from the previous layer
				weights[i - 1][j] = new double[network[i - 1].length];
				// set all weights to 0 initially
				for ( int k = 0 ; k < weights[i-1][j].length ; k++ ) {
					weights[i-1][j][k] = Main.getRandomNumberInRange(-1, 1);                       
				}
			}
		}
	}

	/**
	 * Set weightDeltas to be an array with the same dimension as weights,
	 * with all numbers initialized to 0.
	 */
	public void initializeWeightDeltas() {
		weightDeltas = new double[ weights.length ][][];
		for ( int i = 0 ; i < weights.length ; i++ ) {
			weightDeltas[i] = new double[ weights[i].length ][];
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				weightDeltas[i][j] = new double[ weights[i][j].length ];
				Arrays.fill( weightDeltas[i][j], 0);
			}
		}
	}


	public void initializeTempWeights() {
		tempWeights = cloneWeights( weights );
	}


	public void computeNodeDeltas( double[] targets ) {
		getOutputs();
		for ( int i = network.length - 1 ; i >= 1 ; i-- ) {
			for ( int j = 0 ; j < network[i].length ; j++ ) {
				computeNodeDeltaAtLayer( i, j, targets );
			}
		}
	}

	public void computeNodeDeltaAtLayer( int layerIndex, int nodeIndex, double[] targets ) {
		if ( layerIndex == network.length - 1 ) {
			computeNodeDeltaAtOutputLayer( nodeIndex, targets );
		} else {
			computeNodeDeltaAtHiddenLayer( layerIndex, nodeIndex );
		}
	}


	public void computeNodeDeltaAtOutputLayer( int nodeIndex, double[] targets ) {
		assert( network[network.length - 1].length == targets.length );
		Neuron node = network[ network.length - 1][nodeIndex];
		switch (lossFunctionType) {
			case MSE:
				node.setDelta( 
					(node.output( weights[network.length - 2][nodeIndex] ) - targets[nodeIndex]) * node.getAFDerivative( weights[network.length - 2][nodeIndex] ) 
				);
				break;
			case CROSS_ENTROPY:
				node.setDelta( targets[nodeIndex] - node.output( weights[network.length - 2][nodeIndex] ) );
				break;
			default:
				System.err.println("Error. Undefined loss function");
				break;
		}
	}


	public void computeNodeDeltaAtHiddenLayer( int layerIndex, int nodeIndex ) {
		// this layer cannot be the output layer
		assert( layerIndex < network.length - 1 );
		Neuron node = network[layerIndex][nodeIndex];
		if ( node instanceof BiasNeuron )
			return;
		double sum = 0;
		Neuron[] nextLayer = network[layerIndex + 1];
		for ( int i = 0 ; i < nextLayer.length ; i++ ) {
			Neuron n = nextLayer[i];
			sum += weights[layerIndex][i][nodeIndex] * n.getDelta();
		}
		node.setDelta( node.getAFDerivative( weights[layerIndex-1][nodeIndex] ) * sum );
	}


	/**
	 * Sets the values for the input layer.
	 * @param inputs an array of inputs being passed to the network.
	 */
	public void setInputs( double[] inputs ) {
		for ( int i = 0 ; i < network[0].length; i++ ) {
			// first layer has input neurons
			Neuron n = network[0][i];
			if ( n instanceof InputNeuron ) {
				((InputNeuron) n).setInput( inputs[i] );
			}
		}
	}


	/**
	 * Set the weights for each neuron in the network.
	 * @param weights a 3-dimensional array representing the weights, 
	 * where the indexes are: layer's index, neuron's index, neuron on previous layer's index.
	 */ 
	public void setWeights( double[][][] weights ) {
		// every neuron except input neuron must have an array of weights
		assert( weights.length == network.length - 1 );
		this.weights = weights;
	}


	public void updateWeights() {
		// fill in the tempWeights array
		for ( int i = network.length - 1 ; i >= 1; i-- ) {
			for ( int j = 0 ; j < network[i].length ; j++ ) {
				updateWeightsForNode( i, j );
			}
		}
		// set weight array to tempWeights
		weights = tempWeights;
	}


	/**
	 * Update the weight arrays of a node at a specified position.
	 * <tt>newWeight = oldWeight - weightDelta</tt>
	 */
	public void updateWeightsForNode( int layerIndex, int nodeIndex ) {
		Neuron node = network[layerIndex][nodeIndex];
		if ( node instanceof BiasNeuron ) 
			return;
		for ( int i = 0 ; i < weights[layerIndex-1][nodeIndex].length ; i++ ) {
			Neuron inputNode = network[layerIndex-1][i];
			double gradient = node.getDelta();
			if ( inputNode instanceof BiasNeuron ) {
				continue;
			}
			else if ( inputNode instanceof InputNeuron ) {
				gradient *= ((InputNeuron) inputNode).output();
			}
			else {
				gradient *= inputNode.output( weights[layerIndex-2][i] );
			}
			double prevWeightDelta = weightDeltas[layerIndex-1][nodeIndex][i];
			weightDeltas[layerIndex-1][nodeIndex][i] = 
				learningRate * gradient + momentum * prevWeightDelta;
			double newWeight = weights[layerIndex-1][nodeIndex][i] - weightDeltas[layerIndex-1][nodeIndex][i];
			tempWeights[layerIndex - 1][nodeIndex][i] = newWeight;
			if ( weights[layerIndex-1][nodeIndex][i] == 0.45 ) {
				System.out.println("outttt: " + inputNode.output( weights[layerIndex-2][nodeIndex] ));
				System.out.println( "learning rate " + learningRate );
				System.out.println( "gradient: " + gradient );
				System.out.println( "new weight: " + newWeight );
			}
		}
	}


	public double[][][] getWeights() {
		return weights;
	}


	public void setLossFunctionType( int type ) {
		this.lossFunctionType = type;
	}


	public void setLearningRate( double learningRate ) {
		this.learningRate = learningRate;
	}


	public void setMomentum( double momentum ) {
		this.momentum = momentum;
	}

	public double[][][] cloneWeights( double[][][] weights ) {
		double[][][] cloned = new double[ weights.length ][][];
		for ( int i = 0 ; i < weights.length ; i++ ) {
			cloned[i] = new double[ weights[i].length ][];
			for ( int j = 0 ; j < weights[i].length ; j++ ) {
				cloned[i][j] = new double[ weights[i][j].length ];
				for ( int k = 0 ; k < weights[i][j].length ; k++ ) {
					cloned[i][j][k] = weights[i][j][k];
				}
			}
		}
		return cloned;
	}
}