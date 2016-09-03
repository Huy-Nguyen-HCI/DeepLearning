public class BiasNeuron extends Neuron{

	public BiasNeuron() {
		input = -1; // bias neuron has no input
		activationFunctionType = -1;
	}

	@Override
	public double output() {
		return 1;
	}
}