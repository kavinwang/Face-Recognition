package com.csair.ai.utils;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class NumpyJ {

	public static double[][] expandDims(double[] src, int dim) {
		if(src == null || src.length == 0) {
			return null;
		}
		double[][] dist = null;
		if(dim == 1) {
			dist = new double[src.length][1];
			for(int i = 0; i < src.length; i++) {
				dist[i][0] = src[i];
			}
		}
		return dist;
	}
	
	public static double[][][][] expandDims(double[][][] src, int dim) {
		double[][][][] dist = null;
		if(dim == 0) {
			dist = new double[1][src.length][src[0].length][src[0][0].length];
			for (int i = 0; i < src.length; i++) {
				for (int j = 0; j < src[i].length; j++) {
					for(int k = 0; k < src[i][j].length; k++) {
						dist[0][i][j][k] = src[i][j][k];
					}
				}
			}
		}
		return dist;
	}
	
	public static double[][] hstack(double[]...element) {
		if(element == null || element.length == 0) {
			return null;
		}
		int size = element[0].length, dim = element.length;
		double[][] coords = new double[size][dim];
		for(int i = 0; i < size; i++) {
			for(int j = 0; j < dim; j++) {
				coords[i][j] = element[j][i];
			}
		}
		return coords;
	}
	
	public static double[][] hstack(double[][]...element) {
		if(element == null || element.length == 0) {
			return null;
		}
		int size = element[0].length, dim = 0;
		for(double[][] ele : element) {
			dim += ele[0].length;
		}
		double[][] dist = new double[size][dim];
		for(int i = 0; i < size; i++) {
			int dIndex = 0;
			for(double[][] ele : element) {
				for(int j = 0; j < ele[i].length; j++) {
					dist[i][dIndex++] = ele[i][j];
				}
			}
		}
		return dist;
	}
	
	public static double[][] transpose(double[][] src) {
		double[][] dist = new double[src[0].length][src.length];
		for (int d0 = 0; d0 < src.length; d0++) {
			for (int d1 = 0; d1 < src[d0].length; d1++) {
				dist[d1][d0] = src[d0][d1];
			}
		}
		return dist;
	}
	
	public static double[][][][] transpose1230(double[][][][] src) {
		double[][][][] dist = new double[src[0].length][src[0][0].length][src[0][0][0].length][src.length];
		for (int d0 = 0; d0 < src.length; d0++) {
			for (int d1 = 0; d1 < src[d0].length; d1++) {
				for (int d2 = 0; d2 < src[d0][d1].length; d2++) {
					for (int d3 = 0; d3 < src[d0][d1][d2].length; d3++) {
						dist[d1][d2][d3][d0] = src[d0][d1][d2][d3];
					}
				}
			}
		}
		return dist;
	}
	
	public static double[][][][] transpose0213(double[][][][] src) {
		double[][][][] dist = new double[src.length][src[0][0].length][src[0].length][src[0][0][0].length];
		for (int d0 = 0; d0 < src.length; d0++) {
			for (int d1 = 0; d1 < src[d0].length; d1++) {
				for (int d2 = 0; d2 < src[d0][d1].length; d2++) {
					for (int d3 = 0; d3 < src[d0][d1][d2].length; d3++) {
						dist[d0][d2][d1][d3] = src[d0][d1][d2][d3];
					}
				}
			}
		}
		return dist;
	}

	public static int[] where(double[] src, double threshold, int symbol) {
		if(src == null || src.length == 0) {
			return null;
		}
		List<Integer> items = new ArrayList<Integer>();
		for(int i = 0; i < src.length; i++) {
			if(symbol == 1) {
				if(src[i] >= threshold) {
					items.add(i);
				}
			} else if(symbol == 2) {
				if(src[i] > threshold) {
					items.add(i);
				}
			} else if(symbol == 3) {
				if(src[i] <= threshold) {
					items.add(i);
				}
			} else if(symbol == 4) {
				if(src[i] < threshold) {
					items.add(i);
				}
			}
		}
		if(items.isEmpty()) {
			return null;
		}
		int[] dist = new int[items.size()];
		for(int i = 0; i < items.size(); i++) {
			dist[i] = items.get(i);
		}
		return dist;
	}
	
	public static int[][] where(double[][] src, double threshold, int symbol) {
		if(src == null || src.length == 0) {
			return null;
		}
		List<int[]> items = new ArrayList<int[]>();
		for(int i = 0; i < src.length; i++) {
			for(int j = 0; j < src[i].length; j++) {
				if(symbol == 1) {
					if(src[i][j] >= threshold) {
						items.add(new int[]{i, j});
					}
				} else if(symbol == 2) {
					if(src[i][j] <= threshold) {
						items.add(new int[]{i, j});
					}
				}
			}
		}
		if(items.isEmpty()) {
			return null;
		}
		int[][] dist = new int[items.size()][items.get(0).length];
		for(int i = 0; i < items.size(); i++) {
			dist[i] = items.get(i);
		}
		return dist;
	}
	
	public static double[][] flipud(double[][] src) {
		double[][] dist = new double[src.length][src[0].length];
		int iter = src.length / 2 + 1;
		for (int i = 0; i < iter; i++) {
			dist[i] = src[src.length - i - 1].clone();
			dist[src.length - i -1] = src[i].clone();
		}
		return dist;
	}
	
	public static int[] argsort(double[] src) {
		int[] dist = new int[src.length];
		Map<Double, Integer> map = new TreeMap<Double, Integer>();
		for(int i = 0; i < src.length; i++) {
			map.put(src[i], i);
		}
		Iterator<Double> it = map.keySet().iterator();
		int k = 0;
		while(it.hasNext()) {
			double item = it.next();
			dist[k++] = map.get(item);
		}
		return dist;
	}
	
	public static double[] maximum(double[] src1, double src2) {
		if(src1 == null || src1.length == 0) {
			return null;
		}
		double[] dist = new double[src1.length];
		for(int i = 0; i < src1.length; i++) {
			if(src1[i] >= src2) {
				dist[i] = src1[i];
			} else {
				dist[i] = src2;
			}
		}
		return dist;
	}
	
	public static double[] maximum(double[] src1, double[] src2) {
		if(src1 == null || src1.length == 0 || src2 == null || src2.length == 0) {
			return null;
		}
		double[] dist = new double[src1.length];
		for(int i = 0; i < src1.length; i++) {
			if(src1[i] >= src2[i]) {
				dist[i] = src1[i];
			} else {
				dist[i] = src2[i];
			}
		}
		return dist;
	}
	
	public static double[] minimum(double[] src1, double src2) {
		if(src1 == null || src1.length == 0) {
			return null;
		}
		double[] dist = new double[src1.length];
		for(int i = 0; i < src1.length; i++) {
			if(src1[i] < src2) {
				dist[i] = src1[i];
			} else {
				dist[i] = src2;
			}
		}
		return dist;
	}
	
	public static double[][] fix(double[][] src, int rowb, int rowe, int colb, int cole) {
		if(src == null || src.length == 0) {
			return null;
		}
		double[][] dist = new double[src.length][src[0].length];
		for(int i = 0; i < src.length; i++) {
			for(int j = 0; j < src[i].length; j++) {
				dist[i][j] = src[i][j];
				if(i >= rowb && i < rowe) {
					if(j >= colb && j < cole) {
						if(dist[i][j] >= 0) {
							dist[i][j] = Math.floor(dist[i][j]);
						} else {
							dist[i][j] = Math.ceil(dist[i][j]);
						}
					}
				}
				
			}
		}
		return dist;
	}
	
	public static double[] ones(int size) {
		double[] dist = new double[size];
		for(int i = 0; i < size; i++) {
			dist[i] = 1;
		}
		return dist;
	}
	
}
