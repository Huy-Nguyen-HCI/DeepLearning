package layer;

public class Size {
    int x, y;

    /**
     * Class constructor. Saves the value of the input width and height
     * @param width input width
     * @param height input height
     */
    public Size( int width, int height ) {
        this.x = height;
        this.y = width;
    }

    public int getX() { return x; }

    public int getY() { return y; }
}