import Jama.Matrix;

public class Position {
	Integer layer;
	Integer node;

	public Position( Integer layer, Integer node ) {
		this.layer = layer;
		this.node = node;
	}


	@Override 
	public int hashCode() {
		return layer.hashCode() + node.hashCode();
	}


	@Override 
	public boolean equals( Object o ) {
		if( o == null || !(o instanceof Position) )
   			return false;
		Position p = (Position) o;
		return p.layer.equals(layer) && p.node.equals(node);
	}
}