/**
 * Created by nguyenha on 11/29/2016.
 */
import java.util.*;
import java.io.Serializable;

public class ConvolutionalTraining implements Serializable {

    List<Layer> layers;
    int numberOfLayers;
    int batchSize;
    double learningRate = 1.5;

    /**
     * Class constructor. Sets up convolutional stochastic training for an input network.
     * @param network the network to train on.
     * @param batchSize the size of each batch used in stochastic gradient descent.
     */
    public ConvolutionalTraining( Network network, int batchSize ) {
        layers = network.getLayers();
        numberOfLayers = layers.size();
        this.batchSize = batchSize;
        setup( batchSize );
    }


    /**
     * Sets up each layer based on the batch size.
     * @param batchSize the size of each batch used in stochastic gradient descent.
     */
    private void setup( int batchSize ) {
        Layer inputLayer = layers.get(0);
        // initialize the array of 3D outputs for the input layer
        inputLayer.initOutmaps( batchSize );
        for ( int i = 1 ; i < layers.size() ; i++ ) {
            Layer layer = layers.get(i);
            Layer previous = layers.get(i-1);
            int inputDepth = previous.getOutputDepth();

            switch ( layer.getType() ) {
                case Layer.INPUT:
                    break;
                case Layer.CONV:
                    // set the size of the 3D input
                    // assume stride of 1
                    Size inputSize = previous.getOutputSize();
                    Size filterSize = layer.getFilterSize();
                    Size outputSize = new Size(inputSize.x - filterSize.x + 1, inputSize.y - filterSize.y + 1);
                    layer.setOutputSize( outputSize );
                    layer.initFilters( inputDepth );
                    layer.initBias();
                    layer.initErrors( batchSize );
                    layer.initOutmaps( batchSize );
                    break;
                case Layer.MAXPOOL:
                    // output and input have same depth
                    layer.setOutputDepth( inputDepth );
                    // output size = input size / pooling size
                    Size out = previous.getOutputSize();
                    Size poolingSize = layer.getPoolingSize();
                    layer.setOutputSize( new Size(out.x / poolingSize.x, out.y / poolingSize.y) );
                    layer.initErrors( batchSize );
                    layer.initOutmaps( batchSize );
                    break;
                case Layer.OUTPUT:
                    layer.initOutputFilters( inputDepth, previous.getOutputSize() );
                    layer.initBias();
                    layer.initErrors( batchSize );
                    layer.initOutmaps( batchSize );
                    break;
                default:
                    assert false : "Unrecognized layer type";
                    break;
            }
        }
    }


    /**
     * Train the weights of network.
     * @param trainset the trianing data set.
     * @param numberOfIterations number of training iterations.
     */
    public void train( Dataset trainset, Dataset testset, int numberOfIterations ) {
        for ( int t = 0 ; t < numberOfIterations ; t++ ) {
            int numberOfTrainings = trainset.size() / batchSize;
            int right = 0;
            int count = 0;
            for ( int i = 0 ; i < numberOfTrainings ; i++ ) {
                int[] randPerm = Utilities.randomPerm( trainset.size(), batchSize );
                Layer.prepareForNewBatch();
                for ( int index : randPerm ) {
                    boolean isCorrect = train( trainset.getRecord(index) );
                    if ( isCorrect ) right ++;
                    count ++;
                    Layer.prepareForNewRecord();
                }

                updateParams();
                // decorate the output a bit
                if ( i % 50 == 0 ) {
                    System.out.print("..");
                    if ( i + 50 > numberOfTrainings ) System.out.println();
                }
            }

            double correctPercent = 1.0 * right / count;
            if ( t % 10 == 1 && correctPercent > 0.96 ) {
                learningRate = 0.001 + learningRate * 0.9;
                System.out.println( "new learning rate: " + learningRate );
            }
            System.out.println("Accuracy in training set: " + right + "/" + count + " = " + correctPercent );
//            test( testset );
        }
    }


    /**
     * Train the network for a given record.
     * @param record the input record.
     * @return <tt>true</tt> if the network correctly classifies the record after training and <tt>false</tt> otherwise.
     */
    private boolean train( Record record ) {
        forward( record );
        return backPropagation( record );
    }


    /**
     * Update the parameters.
     */
    private void updateParams() {
        for ( int i = 1 ; i < numberOfLayers ; i++ ) {
            Layer layer = layers.get(i);
            Layer previous = layers.get(i-1);
            int layerType = layer.getType();
            if ( layerType == Layer.CONV || layerType == Layer.OUTPUT ) {
                updateFilters( layer, previous );
                updateBias( layer, previous );
            }
        }
    }


    /**
     * Perform the feedforward cycle when training with a specified record.
     * @param record the input record.
     */
    private void forward(Record record) {
        setInputLayerOutput(record);
        for (int l = 1; l < layers.size(); l++) {
            Layer layer = layers.get(l);
            Layer lastLayer = layers.get(l - 1);
            switch (layer.getType()) {
                case Layer.CONV:
                    setConvOutput(layer, lastLayer);
                    break;
                case Layer.MAXPOOL:
                    setMaxpoolingOutput(layer, lastLayer);
                    break;
                case Layer.OUTPUT:
                    setConvOutput(layer, lastLayer);
                    break;
                default:
                    break;
            }
        }
    }


    private void setInputLayerOutput( Record record ) {
        Layer inputLayer = layers.get(0);
        Size outputSize = inputLayer.getOutputSize();
        double[] attr = record.getAttrs();
        if (attr.length != outputSize.x * outputSize.y)
            throw new RuntimeException("Sizes do not match");
        for (int i = 0; i < outputSize.x; i++) {
            for (int j = 0; j < outputSize.y; j++) {
                inputLayer.setMapValue(0, i, j, attr[outputSize.x * i + j]);
            }
        }
    }


    private void setConvOutput( Layer layer, Layer lastLayer ) {
        int outputDepth = layer.getOutputDepth();
        int lastOutputDepth = lastLayer.getOutputDepth();
        new Task(outputDepth) {

            @Override
            public void process(int start, int end) {
                for (int j = start; j < end; j++) {
                    double[][] sum = null;
                    for (int i = 0; i < lastOutputDepth; i++) {
                        double[][] lastMap = lastLayer.getMap(i);
                        double[][] filter = layer.getFilter(i, j);
                        if (sum == null) {
                            sum = Utilities.convnValid(lastMap, filter);
                        }
                        else {
                            sum = Utilities.addMatrix( Utilities.convnValid(lastMap, filter), sum );
                        }
                    }
                    double bias = layer.getBias(j);
                    // apply sigmoid activation function
                    for ( int row = 0 ; row < sum.length ; row++ ) {
                        for ( int column = 0 ; column < sum[row].length ; column++ ) {
                            sum[row][column] = Utilities.sigmod( sum[row][column] + bias );
                        }
                    }
                    layer.setMapValue( j, sum );
                }
            }

        }.start();
    }


    private void setMaxpoolingOutput( Layer layer, Layer lastLayer ) {
        int lastOutputDepth = lastLayer.getOutputDepth();
        new Task(lastOutputDepth) {

            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    double[][] lastMap = lastLayer.getMap(i);
                    Size poolingSize = layer.getPoolingSize();
                    double[][] sampMatrix = Utilities.scaleMatrix(lastMap, poolingSize);
                    layer.setMapValue(i, sampMatrix);
                }
            }

        }.start();
    }


    /**
     * Perform the backpropagation cycle when training with a specified record.
     * @param record the input record.
     * @return <tt>true</tt> if the network correctly classifies the record after training and <tt>false</tt> otherwise.
     */
    private boolean backPropagation( Record record ) {
        boolean result = setOutputLayerErrors( record );
        setHiddenLayerErrors();
        return result;
    }


    private void updateFilters( Layer layer, Layer previous ) {
        int outputDepth = layer.getOutputDepth();
        int inputDepth = previous.getOutputDepth();
        new Task(outputDepth) {

            @Override
            public void process(int start, int end) {
                for (int j = start; j < end; j++) {
                    for (int i = 0; i < inputDepth; i++) {
                        double[][] weightGradient = null;
                        for (int r = 0; r < batchSize; r++) {
                            double[][] error = layer.getError(r, j);
                            if (weightGradient == null)
                                weightGradient = Utilities.convnValid(previous.getMap(r, i), error);
                            else {
                                weightGradient =
                                        Utilities.addMatrix( weightGradient, Utilities.convnValid(previous.getMap(r, i), error));
                            }
                        }

                        double[][] filter = layer.getFilter(i, j);
                        // calculate the delta matrix then replace it with the new weight
                        for ( int row = 0 ; row < weightGradient.length ; row ++ ) {
                            for ( int column = 0 ; column < weightGradient[row].length ; column ++ ) {
                                weightGradient[row][column] /= batchSize;
                                weightGradient[row][column] =
                                        filter[row][column] + weightGradient[row][column] * learningRate;
                            }
                        }

                        // update the weights
                        layer.setFilter(i, j, weightGradient);
                    }
                }

            }
        }.start();
    }


    private void updateBias( Layer layer, Layer previous ) {
        double[][][][] errors = layer.getErrors();
        int outputDepth = layer.getOutputDepth();

        new Task(outputDepth) {

            @Override
            public void process(int start, int end) {
                for (int j = start; j < end; j++) {
                    double[][] error = Utilities.sum(errors, j);
                    double biasDelta = Utilities.sum(error) / batchSize;
                    double bias = layer.getBias(j) + learningRate * biasDelta;
                    layer.setBias(j, bias);
                }
            }
        }.start();
    }


    private void setHiddenLayerErrors() {
        for ( int i = numberOfLayers - 2 ; i > 0 ; i -- ) {
            Layer layer = layers.get(i);
            Layer nextLayer = layers.get(i+1);
            int type = layer.getType();
            if ( type == Layer.MAXPOOL ) {
                setMaxpoolError( layer, nextLayer );
            }
            else if ( type == Layer.CONV ) {
                setConvError( layer, nextLayer );
            }
        }
    }


    private void setMaxpoolError( Layer layer, Layer nextLayer ) {
        int outputDepth = layer.getOutputDepth();
        int nextOutputDepth = nextLayer.getOutputDepth();
        new Task(outputDepth) {

            @Override
            public void process(int start, int end) {
                for (int i = start; i < end; i++) {
                    double[][] sum = null;
                    for (int j = 0; j < nextOutputDepth; j++) {
                        double[][] nextError = nextLayer.getError(j);
                        double[][] filter = nextLayer.getFilter(i, j);
                        if (sum == null)
                            sum = Utilities.convnFull(nextError, Utilities.rot180(filter));
                        else {
                            sum = Utilities.addMatrix( sum, Utilities.convnFull(nextError, Utilities.rot180(filter)) );
                        }
                    }
                    layer.setError(i, sum);
                }
            }

        }.start();
    }


    private void setConvError( Layer layer, Layer nextLayer ) {
        assert( nextLayer.getType() == Layer.MAXPOOL );
        int outputDepth = layer.getOutputDepth();
        new Task(outputDepth) {

            @Override
            public void process(int start, int end) {
                for (int m = start; m < end; m++) {
                    Size scale = nextLayer.getPoolingSize();
                    double[][] nextError = nextLayer.getError(m);
                    double[][] map = layer.getMap(m);
                    double[][] outMatrix = Utilities.multiplyMatrix(
                            map,
                            Utilities.oneMinusMatrix(Utilities.cloneMatrix(map))
                    );
                    outMatrix = Utilities.multiplyMatrix(
                            outMatrix,
                            Utilities.kronecker(nextError, scale)
                    );
                    layer.setError(m, outMatrix);
                }
            }
        }.start();
    }


    private boolean setOutputLayerErrors( Record record ) {
        Layer outputLayer = layers.get(numberOfLayers - 1);
        int outputDepth = outputLayer.getOutputDepth();
        double[] target = new double[outputDepth];
        double[] outmaps = new double[outputDepth];
        for (int m = 0; m < outputDepth; m++) {
            double[][] outmap = outputLayer.getMap(m);
            outmaps[m] = outmap[0][0];

        }
        int lable = record.getLable().intValue();
        target[lable] = 1;
        for (int m = 0; m < outputDepth; m++) {
            outputLayer.setError(m, 0, 0, outmaps[m] * (1 - outmaps[m]) * (target[m] - outmaps[m]));
        }
        return lable == Utilities.getMaxIndex(outmaps);
    }





    /**
     * Test the accuracy of a network against a given data set.
     * @param dataset the test set.
     * @return the correct percentage when the network predicts the outputs of all inputs from the data set.
     */
    public double test(Dataset trainset) {
        Layer.prepareForNewBatch();
        Iterator<Record> iter = trainset.iter();
        int right = 0;
        while (iter.hasNext()) {
            Record record = iter.next();
            forward(record);
            Layer outputLayer = layers.get(numberOfLayers - 1);
            int mapNum = outputLayer.getOutputDepth();
            double[] out = new double[mapNum];
            for (int m = 0; m < mapNum; m++) {
                double[][] outmap = outputLayer.getMap(m);
                out[m] = outmap[0][0];
            }
            if (record.getLable().intValue() == Utilities.getMaxIndex(out))
                right++;
        }
        double p = 1.0 * right / trainset.size();
        System.out.println("Test accuracy: " + p + "");
        return p;
    }

}