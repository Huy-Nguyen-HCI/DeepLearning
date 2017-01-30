import java.util.Arrays;

/**
 * A record is a tuple of attributes and category
 *
 */
public class Record {
    private double[] attrs;
    private Double label;

    public Record(double[] attrs, Double label) {
        this.attrs = attrs;
        this.label = label;
    }

    public Record(double[] data) {
        attrs = Arrays.copyOfRange(data, 1, data.length);
        label = data[0];
    }

    /**
     * Get the array of attributes.
     *
     * @return
     */
    public double[] getAttrs() {
        return attrs;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("attrs:");
        sb.append(Arrays.toString(attrs));
        sb.append("label:");
        sb.append(label);
        return sb.toString();
    }

    /**
     * Get the category of this record.
     *
     * @return
     */
    public Double getLable() {
        return label;
    }
}