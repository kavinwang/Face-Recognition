package com.csair.ai.vision.facerecognize.core.classifier;

import java.util.Map;

import org.opencv.core.Mat;

import com.csair.ai.tensorflow.TensorflowCaller;
import com.csair.ai.tensorflow.TensorflowServer;
import com.csair.ai.utils.ImageUtils;
import com.csair.ai.utils.NumpyJ;

public class AgeClassifier {
	
	private int inputSize = 227;

	public double[] classify(TensorflowServer server, Mat img) {
		double[] res = null;
		if(img != null) {
			img = ImageUtils.resize(img, inputSize, inputSize);
			double[][][] imDataN = ImageUtils.transChannel(img);
			TensorflowCaller caller = new TensorflowCaller();
			double[][][][] inputs = NumpyJ.expandDims(imDataN, 0);
			
			Map<String, double[][]> tfRes = caller.get2DValue(server, "input", inputs);
			if(tfRes.containsKey("prob")) {
				double[][] probs = tfRes.get("prob");
				if(probs != null && probs.length > 0) {
					res = probs[0];
				}
			}
		}
		return res;
	}
	
}
