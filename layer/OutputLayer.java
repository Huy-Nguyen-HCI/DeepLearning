package layer;

import utilities.Utilities;

/**
 * Created by HuyNguyen on 1/30/17.
 */
public class OutputLayer extends Layer {

    /**
     * Build an output layer.
     * @param the number of categories to classify from
     * @return a layer used as output layer.
     */
    public OutputLayer( int categories ) {
        this.categories = categories;
        this.type = LayerType.OUTPUT;
        this.outputSize = new Size(1,1);
        this.outputDepth = categories; // output is a 1D array, or a 1 x 1 x categories matrix
    }


    public void initOutputFilters( int filterDepth, Size size ) {
        filterSize = size;
        filters = new double[filterDepth][outputDepth][filterSize.x][filterSize.y];
        for ( int i = 0 ; i < filterDepth ; i++ ) {
            for ( int j = 0 ; j < outputDepth ; j++ ) {
                filters[i][j] = Utilities.randomMatrix( filterSize.x, filterSize.y );
            }
        }
    }
}
