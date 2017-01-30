package layer;
import utilities.Utilities;

/**
 * Class that represents a neuron layer in a neural network.
 */
public class Layer {

	public enum LayerType { INPUT, CONV, MAXPOOL, OUTPUT };
	LayerType type;
	int outputDepth; // depth of the output, also number of filters used in this conv layer
	Size filterSize; // the 2D size of each filter
	Size outputSize; // the 2D size of each input map
	double[][][][] filters; // a 4D matrix of weights
	double[] bias; // an array of bias; each filter has one bias
	double[][][][] outmaps; // an array of 3D output matrices for all inputs in one batch
	double[][][][] errors; // an array of 3D error matrices for all inputs in one batch
	static int recordInBatch = 0;  // record the current input index (from 0 to batchSize - 1)
	int categories = -1;     // the number of categories to classify from


	/**
	 * Starting a new batch. Reset the count.
	 */
	public static void prepareForNewBatch() { recordInBatch = 0; }


	/**
	 * Move to the next input in the batch. Update the count.
	 */
	public static void prepareForNewRecord() { recordInBatch ++; }


	/**
	 * Initialize the bias used in this layer.
	 */
	public void initBias() {
		this.bias = Utilities.randomArray( outputDepth );
	}


	/**
	 * Initialize the output matrix array.
	 * @param batchSize the length of the output matrix array.
     */
	public void initOutmaps( int batchSize ) {
		outmaps = new double[batchSize][outputDepth][outputSize.x][outputSize.y];
	}


	// ************** GETTERS AND SETTERS ***************
	public Size getOutputSize() { return outputSize; }

	public void setOutputSize( Size outputSize ) { this.outputSize = outputSize; }

	public LayerType getType() { return type; }

	public int getOutputDepth() { return outputDepth; }

	public void setOutputDepth( int outputDepth ) { this.outputDepth = outputDepth; }

	public void setMapValue( int mapIndex, int x, int y, double value ) {
		outmaps[recordInBatch][mapIndex][x][y] = value;
	}

	public void setMapValue( int mapIndex, double[][] outMatrix ) {
		outmaps[recordInBatch][mapIndex] = outMatrix;
	}

	public double[][] getMap( int index ) {
		return outmaps[recordInBatch][index];
	}

	public void setError( int mapIndex, int x, int y, double value ) {
		errors[recordInBatch][mapIndex][x][y] = value;
	}

	public void setError( int mapIndex, double[][] matrix ) {
		errors[recordInBatch][mapIndex] = matrix;
	}

	public double[][] getError( int mapIndex ) {
		return errors[recordInBatch][mapIndex];
	}

	public double[][][][] getErrors() {
		return errors;
	}

	public void initErrors( int batchSize ) {
		errors = new double[batchSize][outputDepth][outputSize.x][outputSize.y];
	}

	public double getBias( int mapIndex ) {
		return bias[mapIndex];
	}

	public void setBias( int mapIndex, double value ) {
		bias[mapIndex] = value;
	}

	public double[][] getError( int recordIndex, int mapIndex ) {
		return errors[recordIndex][mapIndex];
	}

	public double[][] getMap( int recordIndex, int mapIndex ) {
		return outmaps[recordIndex][mapIndex];
	}

	public double[][] getFilter( int i, int j) {
		return filters[i][j];
	}

	public void setFilter( int i, int j, double[][] mat ) {
		filters[i][j] = mat;
	}

}