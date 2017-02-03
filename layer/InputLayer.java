package layer;

/**
 * Created by HuyNguyen on 1/30/17.
 */
public class InputLayer extends Layer {

    /**
     * Build an input layer.
     * @param outputSize the 2D size of the an input image.
     * @return a layer used as input layer.
     */
    public InputLayer( Size outputSize ) {
        type = LayerType.INPUT;
        // input is one channel of an image (ex: a 28 x 28 x 1 matrix), so output depth is 1
        this.outputDepth = 1;
        setOutputSize( outputSize );
    }
}
