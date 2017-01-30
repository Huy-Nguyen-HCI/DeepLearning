import java.util.List;
import java.util.ArrayList;

/**
 * A network class, which is a placeholder for an array of layers.
 */
public class Network {

    List<Layer> layers = new ArrayList<Layer>();

    public void addLayer( Layer l ) {
        layers.add( l );
    }

    public List<Layer> getLayers() { return layers; }
}