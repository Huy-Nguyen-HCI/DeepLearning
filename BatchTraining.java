public class BatchTraining {

	FullNeuralNetwork network;
	double[][] inputs;
	double[][] targets;

	public BatchTraining( FullNeuralNetwork network, double[][] inputs, double[][] targets ) {
		this.network = network;
		this.inputs = inputs;
		this.targets = targets;
	}


	public void train() {
		// network.initializeWeights();
		network.initializeWeightDeltas();
		network.initializeTempWeights();
		network.setInputs( inputs[0] );
		network.computeNodeDeltas( targets[0] );
		network.updateWeights();
		for ( int i = 0 ; i < network.weights[network.network.length-2].length; i++ ) {
			Main.printArray( network.weights[network.network.length-2][i] );
		}
	}

}