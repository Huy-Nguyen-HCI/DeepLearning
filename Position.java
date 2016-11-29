/**
 * Created by nguyenha on 11/21/2016.
 */
public class Position {
    Integer depth, row, column;

    Position( Integer depth, Integer row, Integer column ) {
        this.depth = depth;
        this.row = row;
        this.column = column;
    }


    @Override
    public boolean equals( Object o ) {
        if ( !(o instanceof Position) ) return false;
        Position pos = (Position) o;
        return depth.equals(pos.depth) && row.equals(pos.row) && column.equals(pos.column);
    }


    @Override
    public int hashCode() {
        return depth.hashCode() + row.hashCode() + column.hashCode();
    }


    @Override
    public String toString() {
        return "(" + depth + "," + row + "," + column + ")";
    }
}
