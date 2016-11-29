import java.util.*;
import Jama.Matrix;
import java.io.*;

public class Utilities {

	static Random rand = new Random();

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


	public static Matrix[] convert1DTo3D( double[] array ) {
		Matrix[] result = new Matrix[1];
		result[0] = new Matrix(1, array.length);
		result[0].setRow( 0, array );
		return result;
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


	public static double[][] readFile( String fileName ) {
		System.out.println("start reading file " + fileName );
		double[][] data = null;
		Scanner sc = null;
		try {
			sc = new Scanner( new File(fileName) );
			data = new double[ countLines(fileName) ][];
			int lineIndex = 0;
			while ( sc.hasNextLine() ) {
				String line = sc.nextLine();
				String[] split = line.split(",");
				data[ lineIndex ] = new double[ split.length ];
				for ( int i = 0 ; i < split.length ; i++ ) {
					data[lineIndex][i] = Double.parseDouble( split[i] );
				}
				lineIndex ++;
			}

		} catch ( IOException e ) {
			e.printStackTrace();
		}
		finally {
			sc.close();
			System.out.println("Finish reading file" + fileName);
		}
		return data;
	}


	public static int countLines(String fileName) throws IOException {
		InputStream is = new BufferedInputStream(new FileInputStream(fileName));
		try {
			byte[] c = new byte[1024];
			int count = 0;
			int readChars = 0;
			boolean empty = true;
			while ((readChars = is.read(c)) != -1) {
				empty = false;
				for (int i = 0; i < readChars; ++i) {
					if (c[i] == '\n') {
						++count;
					}
				}
			}
			return (count == 0 && !empty) ? 1 : count;
		} finally {
			is.close();
		}
	}


	public static boolean compareArrays( double[][] arr1, double[][] arr2 ) {
		if ( arr1.length != arr2.length )
			return false;
		for ( int i = 0 ; i < arr1.length ; i++ ) {
			if ( !Arrays.equals( arr1[i], arr2[i] ) )
				return false;
		}
		return true;
	}


	public static double calculateAccuracy( double[] outputs, double[] targets ) {
		assert( targets.length == outputs.length );
		int numberOfMatches = 0;
		for ( int i = 0 ; i < outputs.length ; i++ ) {
			numberOfMatches += ( outputs[i] == targets[i] ) ? 1 : 0;
		}
		return numberOfMatches * 1.0 / outputs.length;
	}


	public static int[] generateRandomNumbers( int min, int max, int length ) {
		HashSet<Integer> set = new HashSet<Integer>();
		while ( set.size() < length ) {
			set.add( rand.nextInt(max) + min );
		}
		int[] result = new int[length];
		int count = 0;
		for ( int x : set ) {
			result[count++] = x;
		}
		return result;
	}


	public static int findIndexOfMax( double[] arr ) {
		int index = 0;
		double max = arr[0];
		for ( int i = 0 ; i < arr.length ; i++ ) {
			if ( arr[i] > max ) {
				index = i;
				max = arr[i];
			}
		}
		return index;
	}


	public static double findAverage( double[] arr ) {
		double sum = 0;
		for ( double x : arr ) sum += x;
		return sum / arr.length;
	}

}