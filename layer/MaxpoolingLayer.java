package layer;

/**
 * Created by HuyNguyen on 1/30/17.
 */
public class MaxpoolingLayer extends Layer {

    Size poolingSize; // the size of the region to maxpool

    /**
     * Build a maxpooling layer.
     * @param poolingSize the pooling size.
     * @return a layer used as maxpooling layer.
     */
    public MaxpoolingLayer( Size poolingSize ) {
        this.poolingSize = poolingSize;
        this.type = LayerType.MAXPOOL;
    }

    public Size getPoolingSize() { return poolingSize; }
}
