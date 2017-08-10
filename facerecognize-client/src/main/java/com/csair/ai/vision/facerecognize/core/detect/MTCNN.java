package com.csair.ai.vision.facerecognize.core.detect;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.csair.ai.tensorflow.TensorflowCaller;
import com.csair.ai.tensorflow.TensorflowServer;
import com.csair.ai.utils.ArrayUtils;
import com.csair.ai.utils.ImageUtils;
import com.csair.ai.utils.NumpyJ;

public class MTCNN {

	private int maxRecognizeSize = 800; // 最大接受检测的照片大小，主要受检测算法影响
	private int minsize = 20;
	private double factor = 0.709;
	private double[] threshold = { 0.6, 0.7, 0.7 };
	private int margin = 0;

	public int[][] detectFace(Map<String, TensorflowServer> serverMap, Mat img) {
		int height = img.height(), width = img.width();
		double ratio = 1;
		if (Math.max(width, height) > maxRecognizeSize) {
			if (width > height) {
				ratio = (double) maxRecognizeSize / width;
			} else {
				ratio = (double) maxRecognizeSize / height;
			}
			img = ImageUtils.resize(img, (int) (width * ratio), (int) (height * ratio));
		}
		double[][] detectBoxes = detect(serverMap, img);
		int[][] boxes = null;
		if(detectBoxes != null && detectBoxes.length > 0) {
			boxes = new int[detectBoxes.length][4];
			for(int i = 0; i < detectBoxes.length; i++) {
				boxes[i][0] = (int) Math.max(detectBoxes[i][0] / ratio - margin / 2, 0);
				boxes[i][1] = (int) Math.max(detectBoxes[i][1] / ratio - margin / 2, 0);
				boxes[i][2] = (int) Math.min(detectBoxes[i][2] / ratio + margin / 2, width);
				boxes[i][3] = (int) Math.min(detectBoxes[i][3] / ratio + margin / 2, height);
			}
		}
		return boxes;
	}
	
	private double[][] detect(Map<String, TensorflowServer> serverMap, Mat img) {
		if(serverMap == null || img == null) {
			return null;
		}
		double[][] totalBoxes = new double[0][0], points = new double[0][0];
		int w = img.width(), h = img.height();
		double minl = Math.min(h, w), m = (double) 12 / minsize;
		minl = minl * m;
		List<Double> scales = new ArrayList<Double>();
		int factorCount = 0;
		while (minl >= 12) {
			scales.add(m * Math.pow(factor, factorCount));
			minl = minl * factor;
			factorCount += 1;
		}

		TensorflowCaller caller = new TensorflowCaller();
		// first stage
		List<double[]> totalBoxesList = new ArrayList<double[]>();
		for (int i = 0; i < scales.size(); i++) {
			double scale = scales.get(i);
			int hs = (int) Math.ceil(h * scale), ws = (int) Math.ceil(w * scale);
			Mat imData = imresample(img, ws, hs);
			double[][][] imDataN = ImageUtils.normAndTransChannel(imData);
			double[][][][] imDataT = NumpyJ.transpose0213(NumpyJ.expandDims(imDataN, 0));
			Map<String, double[][][][]> pnetRes = caller.get4DValue(serverMap.get("pnet"), "input", imDataT);
			double[][][][] prob = NumpyJ.transpose0213(pnetRes.get("prob")),
					biasadd = NumpyJ.transpose0213(pnetRes.get("biasadd"));
			double[][] boxes = generateBoundingBox(ArrayUtils.cloneLocal4d03(prob, 0, 1),
					ArrayUtils.cloneLocal4d0(biasadd, 0), scale, threshold[0]);

			int[] pick = nms(ArrayUtils.clone(boxes), 0.5, "Union");
			if (boxes != null && boxes.length > 0 && pick != null && pick.length > 0) {
				for (int j = 0; j < pick.length; j++) {
					totalBoxesList.add(ArrayUtils.clone(boxes[pick[j]]));
				}
			}
		}
		if (!totalBoxesList.isEmpty()) {
			totalBoxes = new double[totalBoxesList.size()][totalBoxesList.get(0).length];
			for (int i = 0; i < totalBoxesList.size(); i++) {
				totalBoxes[i] = totalBoxesList.get(i);
			}
		}

		int numbox = totalBoxes.length;
		double[] y = null, ey = null, x = null, ex = null, tmpw = null, tmph = null;
		if (numbox > 0) {
			int[] pick = nms(ArrayUtils.clone(totalBoxes), 0.7, "Union");
			if (pick == null || pick.length == 0) {
				return null;
			}
			totalBoxes = ArrayUtils.pickRows(totalBoxes, pick);
			double[] regw = ArrayUtils.calculate(ArrayUtils.pickCol(totalBoxes, 2), ArrayUtils.pickCol(totalBoxes, 0),
					2),
					regh = ArrayUtils.calculate(ArrayUtils.pickCol(totalBoxes, 3), ArrayUtils.pickCol(totalBoxes, 1),
							2);
			double[][] totalBoxesTemp = new double[pick.length][5];
			for (int i = 0; i < pick.length; i++) {
				totalBoxesTemp[i][0] = totalBoxes[i][0] + totalBoxes[i][5] * regw[i];
				totalBoxesTemp[i][1] = totalBoxes[i][1] + totalBoxes[i][6] * regh[i];
				totalBoxesTemp[i][2] = totalBoxes[i][2] + totalBoxes[i][7] * regw[i];
				totalBoxesTemp[i][3] = totalBoxes[i][3] + totalBoxes[i][8] * regh[i];
				totalBoxesTemp[i][4] = totalBoxes[i][4];
			}
			totalBoxes = totalBoxesTemp;
			totalBoxes = rerec(ArrayUtils.clone(totalBoxes));
			totalBoxes = NumpyJ.fix(totalBoxes, 0, totalBoxes.length, 0, 4);
			Map<String, double[]> tempMap = pad(ArrayUtils.clone(totalBoxes), w, h);
			y = tempMap.get("y");
			ey = tempMap.get("ey");
			x = tempMap.get("x");
			ex = tempMap.get("ex");
			tmpw = tempMap.get("tmpw");
			tmph = tempMap.get("tmph");
		}

		// second stage
		numbox = totalBoxes.length;
		if (numbox > 0) {
			double[][][][] tempimg = new double[numbox][24][24][3];
			for (int k = 0; k < numbox; k++) {
				if ((int) tmph[k] > 0 && (int) tmpw[k] > 0 || (int) tmph[k] == 0 && (int) tmpw[k] == 0) {
					Mat tmpMat = img.rowRange((int) y[k] - 1, (int) ey[k]).colRange((int) x[k] - 1, (int) ex[k]);
					tmpMat = imresample(tmpMat, 24, 24);
					tempimg[k] = ImageUtils.normAndTransChannel(tmpMat);
				} else {
					return null;
				}
			}
			double[][][][] tempimg1 = NumpyJ.transpose0213(tempimg);
			tempimg = NumpyJ.transpose1230(tempimg);
			Map<String, double[][]> rnetRes = caller.get2DValue(serverMap.get("rnet"), "input", tempimg1);
			double[][] prob = rnetRes.get("prob"), conv52 = rnetRes.get("conv52");
			prob = NumpyJ.transpose(prob);
			conv52 = NumpyJ.transpose(conv52);
			double[] score = ArrayUtils.pickRow(prob, 1);
			int[] ipass = NumpyJ.where(score, threshold[1], 2);
			int[] colPick = { 0, 1, 2, 3 };
			if (ipass == null || ipass.length == 0) {
				return null;
			}
			totalBoxes = NumpyJ.hstack(ArrayUtils.pick(totalBoxes, ipass, colPick),
					NumpyJ.expandDims(ArrayUtils.pick(score, ipass), 1));
			double[][] mv = ArrayUtils.pickCols(conv52, ipass);
			if (totalBoxes != null && totalBoxes.length > 0) {
				int[] pick = nms(totalBoxes, 0.7, "Union");
				totalBoxes = ArrayUtils.pickRows(totalBoxes, pick);
				totalBoxes = bbreg(ArrayUtils.clone(totalBoxes), NumpyJ.transpose(ArrayUtils.pickCols(mv, pick)));
				totalBoxes = rerec(ArrayUtils.clone(totalBoxes));
			}
		}

		// third stage
		numbox = totalBoxes.length;
		if (numbox > 0) {
			totalBoxes = NumpyJ.fix(totalBoxes, 0, totalBoxes.length, 0, totalBoxes[0].length);
			Map<String, double[]> tempMap = pad(ArrayUtils.clone(totalBoxes), w, h);
			y = tempMap.get("y");
			ey = tempMap.get("ey");
			x = tempMap.get("x");
			ex = tempMap.get("ex");
			tmpw = tempMap.get("tmpw");
			tmph = tempMap.get("tmph");
			double[][][][] tempimg = new double[numbox][48][48][3];
			for (int k = 0; k < numbox; k++) {
				if ((int) tmph[k] > 0 && (int) tmpw[k] > 0 || (int) tmph[k] == 0 && (int) tmpw[k] == 0) {
					Mat tmpMat = img.rowRange((int) y[k] - 1, (int) ey[k]).colRange((int) x[k] - 1, (int) ex[k]);
					tmpMat = imresample(tmpMat, 48, 48);
					tempimg[k] = ImageUtils.normAndTransChannel(tmpMat);
				} else {
					return null;
				}
			}
			double[][][][] tempimg1 = NumpyJ.transpose0213(tempimg);
			tempimg = NumpyJ.transpose1230(tempimg);
			Map<String, double[][]> onetRes = caller.get2DValue(serverMap.get("onet"), "input", tempimg1);
			double[][] prob = onetRes.get("prob"), conv62 = onetRes.get("conv62"), conv63 = onetRes.get("conv63");
			conv62 = NumpyJ.transpose(conv62);
			conv63 = NumpyJ.transpose(conv63);
			prob = NumpyJ.transpose(prob);
			double[] score = ArrayUtils.pickRow(prob, 1);
			points = conv63;
			int[] ipass = NumpyJ.where(score, threshold[2], 2);
			int[] colPick = { 0, 1, 2, 3 };
			points = ArrayUtils.pickCols(points, ipass);
			if (ipass == null || ipass.length == 0) {
				return null;
			}
			totalBoxes = NumpyJ.hstack(ArrayUtils.pick(totalBoxes, ipass, colPick),
					NumpyJ.expandDims(ArrayUtils.pick(score, ipass), 1));
			double[][] mv = ArrayUtils.pickCols(conv62, ipass);
			double[] bw = ImageUtils.computeDistance(ArrayUtils.pickCol(totalBoxes, 0),
					ArrayUtils.pickCol(totalBoxes, 2)),
					bh = ImageUtils.computeDistance(ArrayUtils.pickCol(totalBoxes, 1),
							ArrayUtils.pickCol(totalBoxes, 3));
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j < points[i].length; j++) {
					points[i][j] = bw[j] * points[i][j] + totalBoxes[j][0] - 1;
				}
			}
			for (int i = 5; i < 10; i++) {
				for (int j = 0; j < points[i].length; j++) {
					points[i][j] = bh[j] * points[i][j] + totalBoxes[j][1] - 1;
				}
			}

			if (totalBoxes != null && totalBoxes.length > 0) {
				totalBoxes = bbreg(ArrayUtils.clone(totalBoxes), NumpyJ.transpose(mv));
				int[] pick = nms(ArrayUtils.clone(totalBoxes), 0.7, "Min");
				totalBoxes = ArrayUtils.pickRows(totalBoxes, pick);
				points = ArrayUtils.pickCols(points, pick);
			}
		}
		return totalBoxes;
	}

	private Mat imresample(Mat img, int ws, int hs) {
		Mat imData = new Mat(hs, ws, CvType.CV_8U);
		Imgproc.resize(img, imData, new Size(ws, hs), 0, 0, Imgproc.INTER_AREA);
		return imData;
	}

	private double[][] generateBoundingBox(double[][] imap, double[][][] reg, double scale, double t) {
		int stride = 2, cellsize = 12;

		imap = NumpyJ.transpose(imap);
		double[][] dx1 = NumpyJ.transpose(ArrayUtils.cloneLocal3d2(reg, 0)),
				dy1 = NumpyJ.transpose(ArrayUtils.cloneLocal3d2(reg, 1)),
				dx2 = NumpyJ.transpose(ArrayUtils.cloneLocal3d2(reg, 2)),
				dy2 = NumpyJ.transpose(ArrayUtils.cloneLocal3d2(reg, 3));
		int[][] points = NumpyJ.where(imap, t, 1);

		if (points == null || points.length == 0) {
			return null;
		} else {
			if (points.length == 1) {
				dx1 = NumpyJ.flipud(dx1);
				dy1 = NumpyJ.flipud(dy1);
				dx2 = NumpyJ.flipud(dx2);
				dy2 = NumpyJ.flipud(dy2);
			}

			double[] score = ArrayUtils.pick(imap, points);
			double[][] coords = NumpyJ.hstack(ArrayUtils.pick(dx1, points), ArrayUtils.pick(dy1, points),
					ArrayUtils.pick(dx2, points), ArrayUtils.pick(dy2, points));

			double[][] q1 = new double[points.length][points[0].length],
					q2 = new double[points.length][points[0].length];
			for (int i = 0; i < points.length; i++) {
				for (int j = 0; j < points[i].length; j++) {
					double q1val = (stride * points[i][j] + 1) / scale,
							q2val = (stride * points[i][j] + cellsize - 1 + 1) / scale;
					if (q1val > 0) {
						q1[i][j] = (int) Math.floor(q1val);
					} else {
						q1[i][j] = (int) Math.ceil(q1val);
					}
					if (q2val > 0) {
						q2[i][j] = (int) Math.floor(q2val);
					} else {
						q2[i][j] = (int) Math.ceil(q2val);
					}
				}
			}

			double[][] boundingBox = NumpyJ.hstack(q1, q2, NumpyJ.expandDims(score, 1), coords);
			return boundingBox;
		}
	}

	private int[] nms(double[][] boxes, double threshold, String method) {
		if (boxes == null || boxes.length == 0) {
			return null;
		}
		double[] x1 = ArrayUtils.pickCol(boxes, 0), y1 = ArrayUtils.pickCol(boxes, 1),
				x2 = ArrayUtils.pickCol(boxes, 2), y2 = ArrayUtils.pickCol(boxes, 3), s = ArrayUtils.pickCol(boxes, 4);
		double[] areas = ImageUtils.computeArea(x1, y1, x2, y2);
		int[] I = NumpyJ.argsort(s), pick = new int[s.length];
		int counter = 0;
		while (I != null && I.length > 0) {
			int i = I[I.length - 1];
			pick[counter] = i;
			counter += 1;
			int[] idx = Arrays.copyOfRange(I, 0, I.length - 1);
			double[] xx1 = NumpyJ.maximum(ArrayUtils.pick(x1, idx), x1[i]),
					yy1 = NumpyJ.maximum(ArrayUtils.pick(y1, idx), y1[i]),
					xx2 = NumpyJ.minimum(ArrayUtils.pick(x2, idx), x2[i]),
					yy2 = NumpyJ.minimum(ArrayUtils.pick(y2, idx), y2[i]);
			double[] w = NumpyJ.maximum(ImageUtils.computeDistance(xx1, xx2), 0.0),
					h = NumpyJ.maximum(ImageUtils.computeDistance(yy1, yy2), 0.0);

			double[] inter = ArrayUtils.calculate(w, h, 3);

			double[] o = null;
			if (method.equals("Min")) {
				o = ArrayUtils.calculate(inter, NumpyJ.minimum(ArrayUtils.pick(areas, idx), areas[i]), 4);
			} else {
				o = ArrayUtils.calculate(inter,
						ArrayUtils.calculate(ArrayUtils.calculate(ArrayUtils.pick(areas, idx), areas[i], 1), inter, 2),
						4);
			}
			I = ArrayUtils.pick(I, NumpyJ.where(o, threshold, 3));
		}
		pick = Arrays.copyOfRange(pick, 0, counter);
		return pick;
	}

	private double[][] rerec(double[][] bboxA) {
		if (bboxA == null || bboxA.length == 0) {
			return null;
		}
		double[] h = ArrayUtils.calculate(ArrayUtils.pickCol(bboxA, 3), ArrayUtils.pickCol(bboxA, 1), 2),
				w = ArrayUtils.calculate(ArrayUtils.pickCol(bboxA, 2), ArrayUtils.pickCol(bboxA, 0), 2);
		double[] l = NumpyJ.maximum(w, h);
		for (int i = 0; i < bboxA.length; i++) {
			bboxA[i][0] = bboxA[i][0] + w[i] * 0.5 - l[i] * 0.5;
			bboxA[i][1] = bboxA[i][1] + h[i] * 0.5 - l[i] * 0.5;
			bboxA[i][2] = bboxA[i][0] + l[i];
			bboxA[i][3] = bboxA[i][1] + l[i];
		}
		return bboxA;
	}

	private Map<String, double[]> pad(double[][] totalBoxes, int w, int h) {
		double[] tmpw = ImageUtils.computeDistance(ArrayUtils.pickCol(totalBoxes, 0),
				ArrayUtils.pickCol(totalBoxes, 2)),
				tmph = ImageUtils.computeDistance(ArrayUtils.pickCol(totalBoxes, 1), ArrayUtils.pickCol(totalBoxes, 3));
		int numbox = totalBoxes.length;

		double[] dx = NumpyJ.ones(numbox), dy = NumpyJ.ones(numbox), edx = ArrayUtils.clone(tmpw),
				edy = ArrayUtils.clone(tmph), x = ArrayUtils.pickCol(totalBoxes, 0),
				y = ArrayUtils.pickCol(totalBoxes, 1), ex = ArrayUtils.pickCol(totalBoxes, 2),
				ey = ArrayUtils.pickCol(totalBoxes, 3);

		int[] tmp = NumpyJ.where(ex, w, 2);
		if (tmp != null && tmp.length > 0) {
			for (int i = 0; i < tmp.length; i++) {
				edx[tmp[i]] = -ex[tmp[i]] + w + tmpw[tmp[i]];
				ex[tmp[i]] = w;
			}
		}

		tmp = NumpyJ.where(ey, h, 2);
		if (tmp != null && tmp.length > 0) {
			for (int i = 0; i < tmp.length; i++) {
				edy[tmp[i]] = -ey[tmp[i]] + h + tmph[tmp[i]];
				ey[tmp[i]] = h;
			}
		}

		tmp = NumpyJ.where(x, 1, 4);
		if (tmp != null && tmp.length > 0) {
			for (int i = 0; i < tmp.length; i++) {
				dx[tmp[i]] = 2 - x[tmp[i]];
				x[tmp[i]] = 1;
			}
		}

		tmp = NumpyJ.where(y, 1, 4);
		if (tmp != null && tmp.length > 0) {
			for (int i = 0; i < tmp.length; i++) {
				dy[tmp[i]] = 2 - y[tmp[i]];
				y[tmp[i]] = 1;
			}
		}

		Map<String, double[]> result = new HashMap<String, double[]>();
		// dy = tempMap.get("dy");
		// edy = tempMap.get("edy");
		// dx = tempMap.get("dx");
		// edx = tempMap.get("edx");
		result.put("y", y);
		result.put("ey", ey);
		result.put("x", x);
		result.put("ex", ex);
		result.put("tmpw", tmpw);
		result.put("tmph", tmph);
		return result;
	}

	private double[][] bbreg(double[][] boundingBox, double[][] reg) {
		if (boundingBox == null || boundingBox.length == 0 || reg == null || reg.length == 0) {
			return null;
		}
		if (reg[0].length == 1) {
			// reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))
		}
		double[] w = ImageUtils.computeDistance(ArrayUtils.pickCol(boundingBox, 0), ArrayUtils.pickCol(boundingBox, 2)),
				h = ImageUtils.computeDistance(ArrayUtils.pickCol(boundingBox, 1), ArrayUtils.pickCol(boundingBox, 3));
		for (int i = 0; i < boundingBox.length; i++) {
			boundingBox[i][0] = boundingBox[i][0] + reg[i][0] * w[i];
			boundingBox[i][1] = boundingBox[i][1] + reg[i][1] * h[i];
			boundingBox[i][2] = boundingBox[i][2] + reg[i][2] * w[i];
			boundingBox[i][3] = boundingBox[i][3] + reg[i][3] * h[i];
		}
		return boundingBox;
	}

}
