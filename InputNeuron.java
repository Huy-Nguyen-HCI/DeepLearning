public class InputNeuron extends Neuron {
	double input;

	public void setInput( double input ) {
		this.input = input;
	}


	public double output() {
		return input;
	}
}