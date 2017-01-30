import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;


public class Dataset {
    // the list of records in this set
    private List<Record> records;
    // indicate the location of the category value
    private int labelIndex;

    private double maxLable = -1;

    public Dataset(int classIndex) {
        this.labelIndex = classIndex;
        records = new ArrayList<Record>();
    }


    public Dataset(List<double[]> datas) {
        this();
        for (double[] data : datas) {
            append(new Record(data));
        }
    }


    private Dataset() {
        this.labelIndex = 0;
        records = new ArrayList<Record>();
    }


    public int size() {
        return records.size();
    }


    public int getLableIndex() {
        return labelIndex;
    }


    public void append(Record record) {
        records.add(record);
    }


    /**
     * Clear the entire set.
     */
    public void clear() {
        records.clear();
    }


    /**
     *  Add a new record to the set.
     *
     * @param attrs the array of attributes
     * @param label the category associated with the input attributes
     */
    public void append(double[] attrs, Double label) {
        records.add(new Record(attrs, label));
    }


    public Iterator<Record> iter() {
        return records.iterator();
    }


    /**
     * Get the attributes at index
     *
     * @param index
     * @return
     */
    public double[] getAttrs(int index) {
        return records.get(index).getAttrs();
    }


    public Double getLable(int index) {
        return records.get(index).getLable();
    }

    /**
     * Load data from an input file
     *
     * @param filePath the path to the input file.
     * @param tag the delimiter used when reading the file.
     * @param labelIndex indicate the location of the label.
     * @return
     */
    public static Dataset load(String filePath, String tag, int labelIndex) {
        Dataset dataset = new Dataset();
        dataset.labelIndex = labelIndex;
        File file = new File(filePath);
        try {

            BufferedReader in = new BufferedReader(new FileReader(file));
            String line;
            while ((line = in.readLine()) != null) {
                String[] datas = line.split(tag);
                if (datas.length == 0)
                    continue;
                double[] data = new double[datas.length];
                for (int i = 0; i < datas.length; i++)
                    data[i] = Double.parseDouble(datas[i]);
                Record record = new Record(data);
                dataset.append(record);
            }
            in.close();

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
        System.out.println("Loading set size:" + dataset.size());
        return dataset;
    }

    /**
     * Get the record at a specified index.
     *
     * @param index the input index.
     * @return the record.
     */
    public Record getRecord(int index) {
        return records.get(index);
    }

}

