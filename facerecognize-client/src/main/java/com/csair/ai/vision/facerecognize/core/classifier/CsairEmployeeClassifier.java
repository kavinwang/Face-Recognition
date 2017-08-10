package com.csair.ai.vision.facerecognize.core.classifier;

import java.util.HashMap;
import java.util.Map;

import org.tensorflow.framework.TensorProto;

import com.csair.ai.tensorflow.TensorflowCaller;
import com.csair.ai.tensorflow.TensorflowClient;
import com.csair.ai.tensorflow.TensorflowServer;

public class CsairEmployeeClassifier {

	public double[] classify(TensorflowServer server, double[] emb) {
		double[] res = null;
		if(emb != null && emb.length > 0) {
			TensorflowCaller tfCaller = new TensorflowCaller();
			double[][] inputs = new double[1][emb.length];
			inputs[0] = emb;
			
			Map<String, TensorProto> response = new HashMap<String, TensorProto>();
			TensorflowClient client = new TensorflowClient(server.getHost(), server.getPort());
			try {
				response = client.inference(server.getModelName(), server.getModelVersion(),
						tfCaller.buildInput("input", inputs));
			} catch(Exception e) {
				e.printStackTrace();
			} finally {
				try {
					client.shutdown();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			double[] scores = null;
			int predict= -1;
			if(response.containsKey("scores")) {
				double[][] resscore = tfCaller.parseResponseD2(response.get("scores"));
				if(resscore != null && resscore.length > 0) {
					scores = resscore[0];
				}
			}
			
			if(response.containsKey("predict")) {
				predict = (int) response.get("predict").getInt64Val(0);
			}
			
			if(predict >= 0) {
				res = new double[2];
				res[0] = predict;
				res[1] = scores[predict];
			}
		}
		return res;
	}
	
}
