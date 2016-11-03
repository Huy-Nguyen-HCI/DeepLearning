import java.util.*;
import jama.Matrix;

public class Utilities {

	public static double[][][] cloneArray( double[][][] array ) {
		double[][][] cloned = new double[ array.length ][][];
		for ( int i = 0 ; i < array.length ; i++ ) {
			cloned[i] = new double[ array[i].length ][];
			for ( int j = 0 ; j < array[i].length ; j++ ) {
				cloned[i][j] = new double[ array[i][j].length ];
				for ( int k = 0 ; k < array[i][j].length ; k++ ) {
					cloned[i][j][k] = array[i][j][k];
				}
			}
		}
		return cloned;
	}


	public static double[] cloneArray( double[] array ) {
		double[] cloned = new double[ array.length ];
		for ( int i = 0 ; i < array.length ; i++ ) {
			cloned[i] = array[i];
		}
		return cloned;
	}


	public static void printArray(double[] arr) {
		System.out.println("*****");
		for ( double x : arr ) System.out.print(x + " ");
		System.out.println("\n*****");
	}


	public static double getRandomNumberInRange( double start, double end ) {
		double random = new Random().nextDouble();
		return start + (random * (end - start));
	}


	public static Matrix create3DBias( double value ) {
		Matrix[] bias = new Matrix[1];
		bias[0] = new Matrix( new double[][]{new double[]{value}} );
		return bias;
	}
}