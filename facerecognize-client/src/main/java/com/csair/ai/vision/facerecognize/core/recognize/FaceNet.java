package com.csair.ai.vision.facerecognize.core.recognize;

import java.util.Map;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.csair.ai.tensorflow.TensorflowCaller;
import com.csair.ai.tensorflow.TensorflowServer;
import com.csair.ai.utils.ImageUtils;

public class FaceNet {

	private int imageSize = 160;

	public double[] calcEmbeddings(TensorflowServer server, Mat image) {
		double[] emb = new double[1];
		TensorflowCaller tfCaller = new TensorflowCaller();
		Mat aligned = new Mat(imageSize, imageSize, CvType.CV_8U);
		Imgproc.resize(image, aligned, new Size(imageSize, imageSize), 0, 0, Imgproc.INTER_LINEAR);

		double[][][] prewhitened = ImageUtils.preWhitenAndTransChannel(aligned);
		double[][][][] images = new double[1][imageSize][imageSize][3];
		images[0] = prewhitened;

		Map<String, double[][]> embMap = tfCaller.get2DValue(server, "input", images);
		double[][] embeddings = embMap.get("embeddings");

		if (embeddings == null || embeddings.length == 0) {
			emb[0] = -1;
		} else {
			emb = embeddings[0];
		}
		return emb;
	}

}
