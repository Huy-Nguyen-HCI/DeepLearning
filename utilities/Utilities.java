package utilities;

import layer.Size;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class Utilities {


	private static Random r = new Random(2);

	public static double[][] addMatrix( double[][] m1, double[][] m2 ) {
		assert m1.length == m2.length && m1[0].length == m2[0].length : "Error: Dimensions must match";
		for ( int i = 0 ; i < m1.length ; i++ ) {
			for ( int j = 0 ; j < m1[i].length ; j++ ) {
				m2[i][j] = m1[i][j] + m2[i][j];
			}
		}
		return m2;
	}


	public static double[][] multiplyMatrix( double[][] m1, double[][] m2 ) {
		double[][] out = new double[m1.length][m1[0].length];
		for ( int i = 0 ; i < m1.length ; i++ ) {
			for ( int j = 0 ; j < m1[i].length ; j++ ) {
				out[i][j] = m1[i][j] * m2[i][j];
			}
		}
		return m2;
	}


	public static double[][] oneMinusMatrix( double[][] m ) {
		for ( int i = 0 ; i < m.length ; i++ ) {
			for ( int j = 0 ; j < m[i].length ; j++ ) {
				m[i][j] = 1 - m[i][j];
			}
		}
		return m;
	}


	public static void printMatrix(double[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			String line = Arrays.toString(matrix[i]);
			line = line.replaceAll(", ", "\t");
			System.out.println(line);
		}
		System.out.println();
	}

	/**
	 * Swap the row and column of a square matrix.
	 *
	 * @param matrix
	 */
	public static double[][] rot180( double[][] matrix ) {
		matrix = cloneMatrix(matrix);
		int m = matrix.length;
		int n = matrix[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n / 2; j++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[i][n - 1 - j];
				matrix[i][n - 1 - j] = tmp;
			}
		}
		for (int j = 0; j < n; j++) {
			for (int i = 0; i < m / 2; i++) {
				double tmp = matrix[i][j];
				matrix[i][j] = matrix[m - 1 - i][j];
				matrix[m - 1 - i][j] = tmp;
			}
		}
		return matrix;
	}

	/**
	 * Generate a random matrix with the specified dimensions.
	 *
	 * @param x the matrix's row
	 * @param y the matrix's column
	 * @return a 2D array of random values.
	 */
	public static double[][] randomMatrix( int x, int y ) {
		double[][] matrix = new double[x][y];
		int tag = 1;
		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				matrix[i][j] = (r.nextDouble() - 0.05) / 10;
			}
		}
		return matrix;
	}

	/**
	 * Generate a random array with specified length.
	 *
	 * @param len the array length
	 * @return an array of random values.
	 */
	public static double[] randomArray(int len) {
		double[] data = new double[len];
		for (int i = 0; i < len; i++) {
			data[i] = (r.nextDouble() - 0.05) / 10;
		}
		return data;
	}

	/**
	 * Generate an array of random indices between 0 and the data set's size.
	 *
	 * @param size the data set's size.
	 * @param batchSize the length of the generated array.
	 * @return an array of indices.
	 */
	public static int[] randomPerm(int size, int batchSize) {
		Set<Integer> set = new HashSet<Integer>();
		while (set.size() < batchSize) {
			set.add(r.nextInt(size));
		}
		int[] randPerm = new int[batchSize];
		int i = 0;
		for (Integer value : set)
			randPerm[i++] = value;
		return randPerm;
	}

	/**
	 * Create a copy of the input matrix.
	 *
	 * @param matrix the input matrix.
	 * @return a copy.
	 */
	public static double[][] cloneMatrix(final double[][] matrix) {

		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				outMatrix[i][j] = matrix[i][j];
			}
		}
		return outMatrix;
	}


	/**
	 * The Kronecker product between the matrix and the pooling map.
	 */
	public static double[][] kronecker(final double[][] matrix, Size scale) {
		final int m = matrix.length;
		int n = matrix[0].length;
		final double[][] outMatrix = new double[m * scale.getX()][n * scale.getY()];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				for (int ki = i * scale.getX(); ki < (i + 1) * scale.getX(); ki++) {
					for (int kj = j * scale.getY(); kj < (j + 1) * scale.getY(); kj++) {
						outMatrix[ki][kj] = matrix[i][j];
					}
				}
			}
		}
		return outMatrix;
	}

	/**
	 * Apply mean sampling on an input matrix.
	 *
	 * @param matrix
	 * @param scaleSize
	 * @return
	 */
	public static double[][] scaleMatrix(final double[][] matrix,
										 final Size scale) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int sm = m / scale.getX();
		final int sn = n / scale.getY();
		final double[][] outMatrix = new double[sm][sn];
		if (sm * scale.getX() != m || sn * scale.getY() != n)
			throw new RuntimeException("scales do not match");
		final int size = scale.getX() * scale.getY();
		for (int i = 0; i < sm; i++) {
			for (int j = 0; j < sn; j++) {
				double sum = 0.0;
				for (int si = i * scale.getX(); si < (i + 1) * scale.getX(); si++) {
					for (int sj = j * scale.getY(); sj < (j + 1) * scale.getY(); sj++) {
						sum += matrix[si][sj];
					}
				}
				outMatrix[i][j] = sum / size;
			}
		}
		return outMatrix;
	}

	/**
	 * Pad the input matrix then perform convolution on it using the input filter.
	 *
	 * @param matrix
	 * @param filter
	 * @return
	 */
	public static double[][] convnFull(double[][] matrix, double[][] filter) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = filter.length;
		final int kn = filter[0].length;
		// À©Õ¹¾ØÕó
		final double[][] extendMatrix = new double[m + 2 * (km - 1)][n + 2
				* (kn - 1)];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				extendMatrix[i + km - 1][j + kn - 1] = matrix[i][j];
		}
		return convnValid(extendMatrix, filter);
	}

	/**
	 * Perform convolution on the input matrix using the input filter.
	 *
	 * @param matrix
	 * @param filter
	 * @return
	 */
	public static double[][] convnValid(final double[][] matrix,
										double[][] filter) {
		int m = matrix.length;
		int n = matrix[0].length;
		final int km = filter.length;
		final int kn = filter[0].length;
		int kns = n - kn + 1;
		final int kms = m - km + 1;
		// ½á¹û¾ØÕó
		final double[][] outMatrix = new double[kms][kns];

		for (int i = 0; i < kms; i++) {
			for (int j = 0; j < kns; j++) {
				double sum = 0.0;
				for (int ki = 0; ki < km; ki++) {
					for (int kj = 0; kj < kn; kj++)
						sum += matrix[i + ki][j + kj] * filter[ki][kj];
				}
				outMatrix[i][j] = sum;

			}
		}
		return outMatrix;

	}

	/**
	 * Calculate the sum of all values in a matrix.
	 *
	 * @param error
	 * @return
	 */

	public static double sum(double[][] matrix) {
		int m = matrix.length;
		int n = matrix[0].length;
		double sum = 0.0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				sum += matrix[i][j];
			}
		}
		return sum;
	}

	/**
	 * Sum all the 2D matrices at column j.
	 *
	 * @param errors the 4D matrix of errors.
	 * @param j the column to sum.
	 * @return a 2D matrix that is the sum of all matrices at column j.
	 */
	public static double[][] sum(double[][][][] errors, int j) {
		int m = errors[0][j].length;
		int n = errors[0][j][0].length;
		double[][] result = new double[m][n];
		for (int mi = 0; mi < m; mi++) {
			for (int nj = 0; nj < n; nj++) {
				double sum = 0;
				for (int i = 0; i < errors.length; i++) {
					sum += errors[i][j][mi][nj];
				}
				result[mi][nj] = sum;
			}
		}
		return result;
	}


	/**
	 * Get the index of the maximum number in an array.
	 *
	 * @param out
	 * @return
	 */
	public static int getMaxIndex(double[] out) {
		double max = out[0];
		int index = 0;
		for (int i = 1; i < out.length; i++)
			if (out[i] > max) {
				max = out[i];
				index = i;
			}
		return index;
	}
}