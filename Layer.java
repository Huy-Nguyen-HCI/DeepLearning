/**
 * Class that represents a neuron layer in a neural network.
 */
public class Layer {

	public static final int INPUT = 0, CONV = 1, MAXPOOL = 2, OUTPUT = 3;
	int type;
	int numberOfFilters; // number of filters used in this layer
	Size outputSize; // the 2D size of each input map
	Size filterSize; // the 2D size of each filter
	Size poolingSize; // the size of the region to maxpool
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
	 * Build an input layer.
	 * @param outputSize the 2D size of the an input image.
	 * @return a layer used as input layer.
	 */
	public static Layer buildInputLayer( Size outputSize ) {
		Layer layer = new Layer();
		layer.type = Layer.INPUT;
		// input is one channel of an image (ex: a 28 x 28 x 1 matrix), so numberOfFilters is 1
		layer.numberOfFilters = 1;
		layer.setOutputSize( outputSize );
		return layer;
	}


	/**
	 * Build a convolutional layer.
	 * @param numberOfFilters the number of filters being used.
	 * @param filterSize the 2D size of a filter.
	 * @return a layer used as convolutional layer.
	 */
	public static Layer buildConvLayer( int numberOfFilters, Size filterSize ) {
		Layer layer = new Layer();
		layer.type = Layer.CONV;
		layer.numberOfFilters = numberOfFilters;
		layer.filterSize = filterSize;
		return layer;
	}


	/**
	 * Build a maxpooling layer.
	 * @param poolingSize the pooling size.
	 * @return a layer used as maxpooling layer.
	 */
	public static Layer buildMaxpoolLayer( Size poolingSize ) {
		Layer layer = new Layer();
		layer.poolingSize = poolingSize;
		layer.type = Layer.MAXPOOL;
		return layer;
	}

	/**
	 * Build an output layer.
	 * @param the number of categories to classify from
	 * @return a layer used as output layer.
	 */
	public static Layer buildOutputLayer( int categories ) {
		Layer layer = new Layer();
		layer.categories = categories;
		layer.type = Layer.OUTPUT;
		layer.outputSize = new Size(1,1);
		layer.numberOfFilters = categories; // output is a 1D array, or a 1 x 1 x categories matrix
		return layer;
	}


	public void initFilters( int filterDepth ) {
		filters = new double[filterDepth][numberOfFilters][filterSize.x][filterSize.y];
		for ( int i = 0 ; i < filterDepth ; i++ ) {
			for ( int j = 0 ; j < numberOfFilters ; j++ ) {
				filters[i][j] = Utilities.randomMatrix( filterSize.x, filterSize.y );
			}
		}
	}


	public void initOutputFilters( int filterDepth, Size size ) {
		filterSize = size;
		filters = new double[filterDepth][numberOfFilters][filterSize.x][filterSize.y];
		for ( int i = 0 ; i < filterDepth ; i++ ) {
			for ( int j = 0 ; j < numberOfFilters ; j++ ) {
				filters[i][j] = Utilities.randomMatrix( filterSize.x, filterSize.y );
			}
		}
	}

	public void initBias() {
		this.bias = Utilities.randomArray( numberOfFilters );
	}


	public void initOutmaps( int batchSize ) {
		outmaps = new double[batchSize][numberOfFilters][outputSize.x][outputSize.y];
	}

	// ************** GETTERS AND SETTERS ***************
	public Size getOutputSize() { return outputSize; }

	public void setOutputSize( Size outputSize ) { this.outputSize = outputSize; }

	public int getType() { return type; }

	public int getOutputDepth() { return numberOfFilters; }

	public void setOutputDepth( int numberOfFilters ) { this.numberOfFilters = numberOfFilters; }

	public Size getFilterSize() { return filterSize; }

	public Size getPoolingSize() { return poolingSize; }

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
		errors = new double[batchSize][numberOfFilters][outputSize.x][outputSize.y];
	}

	public double[][] get2DFilter( int index, int filterIndex ) {
		return filters[index][filterIndex];
	}

	public void set2DFilter( int mapIndex, int filterIndex, double[][] filter ) {
		filters[mapIndex][filterIndex] = filter;
	}

	public double getBias( int mapIndex ) {
		return bias[mapIndex];
	}

	public void setBias( int mapIndex, double value ) {
		bias[mapIndex] = value;
	}

	public double[][][][] getMaps() {
		return outmaps;
	}

	public double[][] getError( int recordIndex, int mapIndex ) {
		return errors[recordIndex][mapIndex];
	}

	public double[][] getMap( int recordIndex, int mapIndex ) {
		return outmaps[recordIndex][mapIndex];
	}

	public int getNumberOfCategories() {
		return categories;
	}

	public double[][][][] getFilter() {
		return filters;
	}

	public double[][] getFilter( int i, int j) {
		return filters[i][j];
	}

	public void setFilter( int i, int j, double[][] mat ) {
		filters[i][j] = mat;
	}

}