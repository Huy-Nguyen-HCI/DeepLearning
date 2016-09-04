public class BiasNeuron extends Neuron{

	double bias;

	public BiasNeuron( double bias ) {
		activationFunctionType = -1;
		this.bias = bias;
	}

	@Override
	public double output() {
		return bias;
	}
}