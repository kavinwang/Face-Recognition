package com.csair.ai.utils;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImageUtils {

	public static double[] computeArea(double[] x1, double[] y1, double[] x2, double[] y2) {
		double[] areas = new double[x1.length];
		for (int i = 0; i < x1.length; i++) {
			areas[i] = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1);
		}
		return areas;
	}

	public static double[] computeDistance(double[] d1, double[] d2) {
		if (d1 == null || d1.length == 0 || d2 == null || d2.length == 0) {
			return null;
		}
		double[] dis = new double[d1.length];
		for (int i = 0; i < d1.length; i++) {
			dis[i] = d2[i] - d1[i] + 1;
		}
		return dis;
	}
	
	public static double[][][] transChannel(Mat img) {
		double[][][] imgN = new double[img.height()][img.width()][3];
		for (int i = 0; i < img.height(); i++) {
			for (int j = 0; j < img.width(); j++) {
				imgN[i][j][0] = img.get(i, j)[2];
				imgN[i][j][1] = img.get(i, j)[1];
				imgN[i][j][2] = img.get(i, j)[1];
			}
		}
		return imgN;
	}
	
	public static double[][][] normAndTransChannel(Mat img) {
		double[][][] imgN = new double[img.height()][img.width()][3];
		for (int i = 0; i < img.height(); i++) {
			for (int j = 0; j < img.width(); j++) {
				imgN[i][j][0] = (img.get(i, j)[2] - 127.5) * 0.0078125;
				imgN[i][j][1] = (img.get(i, j)[1] - 127.5) * 0.0078125;
				imgN[i][j][2] = (img.get(i, j)[1] - 127.5) * 0.0078125;
			}
		}
		return imgN;
	}
	
	public static double[][][] preWhitenAndTransChannel(Mat img) {
		if(img == null) {
			return null;
		}
		double[][][] imgW = new double[img.height()][img.width()][3];
		MatOfDouble meanMat = new MatOfDouble(), stdMat = new MatOfDouble();
		Size size = img.size();
		Core.meanStdDev(img, meanMat, stdMat);
		double mean = meanMat.get(0,0)[0], std = stdMat.get(0,0)[0];
		std = Math.max(std, 1.0 / Math.sqrt(size.height * size.width * img.channels()));
		
		for (int i = 0; i < img.height(); i++) {
			for (int j = 0; j < img.width(); j++) {
				imgW[i][j][0] = (img.get(i, j)[2] - mean) / std;
				imgW[i][j][1] = (img.get(i, j)[1] - mean) / std;
				imgW[i][j][2] = (img.get(i, j)[1] - mean) / std;
			}
		}
		return imgW;
	}
	
	public static Mat resize(Mat src, int width, int height) {
		Mat dist = new Mat(width, height, CvType.CV_8U);
		Imgproc.resize(src, dist, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
		return dist;
	}
	
	public static Mat rotate90n(Mat src, int angle) {
		Mat dst = new Mat();
		if(angle <= 90) {
			Core.transpose(src, dst);
			Core.flip(dst, dst, 1);
		} else if(angle <= 180) {
			Core.flip(src, dst, -1);
		} else if(angle <= 270) {
			Core.transpose(src, dst);
			Core.flip(dst, dst, 0);
		} else {
			src.copyTo(dst);
		}
		return dst;
	}
	
}
