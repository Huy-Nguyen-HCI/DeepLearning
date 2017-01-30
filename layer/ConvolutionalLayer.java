package layer;

import utilities.Utilities;

/**
 * Created by HuyNguyen on 1/30/17.
 */
public class ConvolutionalLayer extends Layer {

    /**
     * Build a convolutional layer.
     * @param outputDepth the number of filters being used.
     * @param filterSize the 2D size of a filter.
     * @return a layer used as convolutional layer.
     */
    public ConvolutionalLayer( int outputDepth, Size filterSize ) {
        type = LayerType.CONV;
        this.outputDepth = outputDepth;
        this.filterSize = filterSize;
    }

    public void initFilters( int filterDepth ) {
        filters = new double[filterDepth][outputDepth][filterSize.x][filterSize.y];
        for ( int i = 0 ; i < filterDepth ; i++ ) {
            for ( int j = 0 ; j < outputDepth ; j++ ) {
                filters[i][j] = Utilities.randomMatrix( filterSize.x, filterSize.y );
            }
        }
    }

    public Size getFilterSize() { return filterSize; }
}
