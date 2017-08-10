package com.csair.ai.vision.facerecognize.service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.csair.ai.tensorflow.TensorflowServer;
import com.csair.ai.utils.ImageUtils;
import com.csair.ai.vision.facerecognize.common.PropertyLoader;
import com.csair.ai.vision.facerecognize.core.detect.MTCNN;
import com.csair.ai.vision.facerecognize.model.DetectionWindow;

@Service("faceDetectService")
@Transactional
public class FaceDetectService {

	@Autowired
	private PropertyLoader propertyLoader;
	private static Map<String, TensorflowServer> serverMap;
	
	public List<DetectionWindow> mtcnn(byte[] imgByte, int mode) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat img = Imgcodecs.imdecode(new MatOfByte(imgByte),
				Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
		if (mode == 1) {
			img = ImageUtils.rotate90n(img, 90);
		}
		initTFServerMap();
		MTCNN mtcnn = new MTCNN();
		int[][] totalBoxes = mtcnn.detectFace(serverMap, img);
		List<DetectionWindow> windows = new ArrayList<DetectionWindow>();
		if (totalBoxes != null && totalBoxes.length > 0) {
			for (int[] box : totalBoxes) {
				windows.add(new DetectionWindow(box));
			}
		}
		return windows;
	}
	
	private void initTFServerMap() {
		if (serverMap == null || serverMap.isEmpty()) {
			serverMap = new HashMap<String, TensorflowServer>();
			serverMap.put("pnet",
					new TensorflowServer(propertyLoader.getStringProperty("mtcnn.pnet.host"),
							propertyLoader.getIntegerProperty("mtcnn.pnet.port"),
							propertyLoader.getStringProperty("mtcnn.pnet.model.name"),
							propertyLoader.getIntegerProperty("mtcnn.pnet.model.v")));
			serverMap.put("rnet",
					new TensorflowServer(propertyLoader.getStringProperty("mtcnn.rnet.host"),
							propertyLoader.getIntegerProperty("mtcnn.rnet.port"),
							propertyLoader.getStringProperty("mtcnn.rnet.model.name"),
							propertyLoader.getIntegerProperty("mtcnn.rnet.model.v")));
			serverMap.put("onet",
					new TensorflowServer(propertyLoader.getStringProperty("mtcnn.onet.host"),
							propertyLoader.getIntegerProperty("mtcnn.onet.port"),
							propertyLoader.getStringProperty("mtcnn.onet.model.name"),
							propertyLoader.getIntegerProperty("mtcnn.onet.model.v")));
		}
	}
	
}
