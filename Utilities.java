import java.util.*;
import Jama.Matrix;

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


	public static void printArray(double[][][] arr) {
		Matrix[] mats = new Matrix[arr.length];
		for ( int i = 0 ; i < arr.length ; i++ ) {
			mats[i] = new Matrix( arr[i] );
		}
		print3DMatrix( mats );
	}


	public static double getRandomNumberInRange( double start, double end ) {
		double random = new Random().nextDouble();
        random = Math.round(random * 100.0) / 100.0;
        return start + (random * (end - start));
	}


	public static Matrix[] create3DBias( double value ) {
		Matrix[] bias = new Matrix[1];
		bias[0] = new Matrix( new double[][]{new double[]{value}} );
		return bias;
	}



	public static Matrix[] createMatrixWithSameDimension( Matrix[] input ) {
		Matrix[] output = new Matrix[input.length];
		for ( int k = 0 ; k < input.length ; k++ ) {
			if ( input[k] == null ) System.out.println(k);
			output[k] = new Matrix( input[k].getRowDimension(), input[k].getColumnDimension() );
		}
		return output;
	}


	public static boolean matricesHaveSameDimension( Matrix[] a, Matrix[] b ) {
		if ( a.length != b.length ) return false;
		for ( int i = 0 ; i < a.length ; i++ ) {
			if (a[i].getRowDimension() != b[i].getRowDimension() ||
					a[i].getColumnDimension() != b[i].getColumnDimension())
				return false;
		}
		return true;
	}


	public static void print3DMatrix( Matrix[] matrix ) {
		for ( int k = 0 ; k < matrix.length ; k++ ) {
			for ( int i = 0 ; i < matrix[k].getRowDimension() ; i++ ) {
				for ( int j = 0 ; j < matrix[k].getColumnDimension() ; j++ ) {
					System.out.print( matrix[k].get(i,j) + " " );
				}
				System.out.println();
			}
			System.out.println("\n\n");
		}
	}
}