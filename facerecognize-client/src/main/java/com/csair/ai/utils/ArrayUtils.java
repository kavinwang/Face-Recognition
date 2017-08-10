package com.csair.ai.utils;

import java.util.Arrays;

public class ArrayUtils {

	public static int[] pick(int[] src, int[] index) {
		if (src == null || index == null || index.length == 0) {
			return null;
		}
		int[] dist = new int[index.length];
		for (int i = 0; i < index.length; i++) {
			dist[i] = src[index[i]];
		}
		return dist;
	}

	public static double[] pick(double[] src, int[] index) {
		if (src == null || index == null || index.length == 0) {
			return null;
		}
		double[] dist = new double[index.length];
		for (int i = 0; i < index.length; i++) {
			dist[i] = src[index[i]];
		}
		return dist;
	}

	public static double[] pick(double[][] src, int[][] index) {
		if (src == null || src.length == 0 || index == null || index.length == 0) {
			return null;
		}
		double[] dist = new double[index.length];
		for (int i = 0; i < index.length; i++) {
			dist[i] = src[index[i][0]][index[i][1]];
		}
		return dist;
	}

	public static double[][] pick(double[][] src, int[] rows, int[] cols) {
		if (src == null || src.length == 0 || rows == null || rows.length == 0 || cols == null || cols.length == 0) {
			return null;
		}
		double [][] dist = new double[rows.length][cols.length];
		for(int i = 0; i < rows.length; i++) {
			for(int j = 0; j < cols.length; j++) {
				dist[i][j] = src[rows[i]][cols[j]];
			}
		}
		return dist;
	}

	public static double[] pickCol(double[][] src, int col) {
		double[] dist = new double[src.length];
		for (int i = 0; i < src.length; i++) {
			dist[i] = src[i][col];
		}
		return dist;
	}

	public static double[] pickRow(double[][] src, int row) {
		if (src == null || src.length == 0) {
			return null;
		}
		double[] dist = new double[src[row].length];
		for (int i = 0; i < src[row].length; i++) {
			dist[i] = src[row][i];
		}
		return dist;
	}

	public static double[][] pickRows(double[][] src, int[] rows) {
		if (src == null || src.length == 0 || rows == null || rows.length == 0) {
			return null;
		}
		double[][] dist = new double[rows.length][src[0].length];
		for (int i = 0; i < rows.length; i++) {
			dist[i] = Arrays.copyOf(src[rows[i]], src[rows[i]].length);
		}
		return dist;
	}
	
	public static double[][] pickCols(double[][] src, int[] cols) {
		if (src == null || src.length == 0 || cols == null || cols.length == 0) {
			return null;
		}
		double[][] dist = new double[src.length][cols.length];
		for (int i = 0; i < src.length; i++) {
			for(int j = 0; j < cols.length; j++) {
				dist[i][j] = src[i][cols[j]];
			}
		}
		return dist;
	}

	public static double[] clone(double[] src) {
		if (src == null || src.length == 0) {
			return null;
		}
		return Arrays.copyOf(src, src.length);
	}

	public static double[][] clone(double[][] src) {
		if (src == null || src.length == 0) {
			return null;
		}
		double[][] dist = new double[src.length][src[0].length];
		for (int d0 = 0; d0 < src.length; d0++) {
			for (int d1 = 0; d1 < src[d0].length; d1++) {
				dist[d0][d1] = src[d0][d1];
			}
		}
		return dist;
	}

	public static double[][][][] clone(double[][][][] src) {
		if (src == null || src.length == 0) {
			return null;
		}
		double[][][][] dist = new double[src.length][src[0].length][src[0][0].length][src[0][0][0].length];
		for (int d0 = 0; d0 < src.length; d0++) {
			for (int d1 = 0; d1 < src[d0].length; d1++) {
				for (int d2 = 0; d2 < src[d0][d1].length; d2++) {
					for (int d3 = 0; d3 < src[d0][d1][d2].length; d3++) {
						dist[d0][d1][d2][d3] = src[d0][d1][d2][d3];
					}
				}
			}
		}
		return dist;
	}

	public static double[][] cloneLocal3d2(double[][][] src, int dim2) {
		double[][] dist = new double[src.length][src[0].length];
		for (int d0 = 0; d0 < src.length; d0++) {
			for (int d1 = 0; d1 < src[d0].length; d1++) {
				dist[d0][d1] = src[d0][d1][dim2];
			}
		}
		return dist;
	}

	public static double[][] cloneLocal4d03(double[][][][] src, int dim0, int dim3) {
		double[][] dist = new double[src[0].length][src[0][0].length];
		for (int d1 = 0; d1 < src[dim0].length; d1++) {
			for (int d2 = 0; d2 < src[dim0][d1].length; d2++) {
				dist[d1][d2] = src[dim0][d1][d2][dim3];
			}
		}
		return dist;
	}

	public static double[][][] cloneLocal4d0(double[][][][] src, int dim0) {
		double[][][] dist = new double[src[0].length][src[0][0].length][src[0][0][0].length];
		for (int d1 = 0; d1 < src[dim0].length; d1++) {
			for (int d2 = 0; d2 < src[dim0][d1].length; d2++) {
				for (int d3 = 0; d3 < src[dim0][d1][d2].length; d3++) {
					dist[d1][d2][d3] = src[dim0][d1][d2][d3];
				}
			}
		}
		return dist;
	}

	public static double[] calculate(double[] a1, double a2, int symbol) {
		if (a1 == null || a1.length == 0) {
			return null;
		}
		double[] dist = new double[a1.length];
		for (int i = 0; i < a1.length; i++) {
			if (symbol == 1) {
				dist[i] = a1[i] + a2;
			} else if (symbol == 2) {
				dist[i] = a1[i] - a2;
			} else if (symbol == 3) {
				dist[i] = a1[i] * a2;
			} else if (symbol == 4) {
				dist[i] = a1[i] / a2;
			}
		}
		return dist;
	}

	public static double[] calculate(double[] a1, double[] a2, int symbol) {
		if (a1 == null || a1.length == 0 || a2 == null || a2.length == 0) {
			return null;
		}
		double[] dist = new double[a1.length];
		for (int i = 0; i < a1.length; i++) {
			if (symbol == 1) {
				dist[i] = a1[i] + a2[i];
			} else if (symbol == 2) {
				dist[i] = a1[i] - a2[i];
			} else if (symbol == 3) {
				dist[i] = a1[i] * a2[i];
			} else if (symbol == 4) {
				dist[i] = a1[i] / a2[i];
			}
		}
		return dist;
	}

}
