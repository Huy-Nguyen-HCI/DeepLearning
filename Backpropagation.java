
public class Backpropagation {

	FullNeuralNetwork network;
	double learningRate = 0.5, momentum = 0;
	final int numberOfIterations = 1000000;
	double[][] inputs;
	double[][] targets;

	final int 
		MSE = 0,
		CROSS_ENTROPY = 1;
	// types of loss function
	int lossFunctionType = MSE;


	public Backpropagation( FullNeuralNetwork network, double[][] inputs, double[][] targets ) {
		this.network = network;
		this.inputs = inputs;
		this.targets = targets;
	}



	public void train() {
		for ( int j = 0 ; j < numberOfIterations ; j++ ) {
			for ( int i = 0 ; i < inputs.length ; i++ ) {
				iterate( inputs[i], targets[i] );
			}
			displayLoss();
		}
		System.out.println("final output is :" );
		for ( double[] input : inputs ) {
			network.setInputs( input );
			Utilities.printArray( network.getOutputs() );
		}

	}


	public void iterate( double[] inputs, double[] targets ) {
		network.setInputs( inputs );
		computeNodeDeltas( targets );
		updateWeights();
	}


	public void computeNodeDeltas( double[] targets ) {
		network.getOutputs();
		for ( int i = network.getNumberOfLayers() - 1 ; i >= 1 ; i-- ) {
			for ( int j = 0 ; j < network.getLayer(i).length ; j++ ) {
				computeNodeDeltaAtLayer( i, j, targets );
			}
		}
	}


	public void computeNodeDeltaAtLayer( int layerIndex, int nodeIndex, double[] targets ) {
		if ( layerIndex == network.getNumberOfLayers() - 1 ) {
			computeNodeDeltaAtOutputLayer( nodeIndex, targets );
		} else {
			computeNodeDeltaAtHiddenLayer( layerIndex, nodeIndex );
		}
	}


	public void computeNodeDeltaAtOutputLayer( int nodeIndex, double[] targets  ) {
		Neuron node = network.getNode( network.getNumberOfLayers() - 1, nodeIndex );
		switch (lossFunctionType) {
			case MSE:
				node.setDelta( 
					(node.output() - targets[nodeIndex]) * node.getAFDerivative() 
				);
				break;
			case CROSS_ENTROPY:
				node.setDelta( targets[nodeIndex] - node.output() );
				break;
			default:
				assert false : "Error. Unrecognized loss function.";
		}
	}


	public void computeNodeDeltaAtHiddenLayer( int layerIndex, int nodeIndex ) {
		// this layer cannot be the output layer
		assert( layerIndex < network.getNumberOfLayers() - 1 );
		Neuron node = network.getNode( layerIndex, nodeIndex );
		if ( node instanceof BiasNeuron )
			return;
		double sum = 0;
		Neuron[] nextLayer = network.getLayer( layerIndex + 1 );
		for ( int i = 0 ; i < nextLayer.length ; i++ ) {
			Neuron n = nextLayer[i];
			sum += n.getWeight( nodeIndex ) * n.getDelta();
		}
		node.setDelta( node.getAFDerivative() * sum );
	}


	public void updateWeights() {
		// update temp weights
		for ( int i = network.getNumberOfLayers() - 1 ; i >= 1; i-- ) {
			for ( int j = 0 ; j < network.getLayer(i).length ; j++ ) {
				updateNewWeightsForNode( i, j );
			}
		}
		// after all temp weights are evaluated, update weights
		for ( int i = network.getNumberOfLayers() - 1 ; i >= 1; i-- ) {
			for ( int j = 0 ; j < network.getLayer(i).length ; j++ ) {
				Neuron n = network.getNode( i, j );
				n.updateWeights();
			}
		}
	}


	/**
	 * Update the weight arrays of a node at a specified position.
	 * <tt>newWeight = oldWeight - weightDelta</tt>
	 */
	public void updateNewWeightsForNode( int layerIndex, int nodeIndex ) {
		Neuron node = network.getNode( layerIndex, nodeIndex );
		if ( node instanceof BiasNeuron ) 
			return;
		for ( int i = 0 ; i < node.getWeights().length ; i++ ) {
			Neuron inputNode = network.getNode( layerIndex - 1, i );
			double gradient = node.getDelta();
			if ( inputNode instanceof BiasNeuron ) {
				continue;
			}
			gradient *= inputNode.output();
			double prevWeightDelta = node.getWeightDelta(i);
			node.setWeightDelta( i, learningRate * gradient + momentum * prevWeightDelta );
			double newWeight = node.getWeight(i) - node.getWeightDelta(i);
			node.setTempWeight( i, newWeight );
		}
	}


	public void displayLoss() {
		double[] out = new double[inputs.length];
		double[] expected = new double[inputs.length];
		for ( int i = 0 ; i < inputs.length ; i++ ) {
			network.setInputs( inputs[i] );
			out[i] = network.getOutputs()[0];
			expected[i] = targets[i][0];
		}
		System.out.println("loss: " + LossFunction.meanSquareError(out, expected));
	}

}