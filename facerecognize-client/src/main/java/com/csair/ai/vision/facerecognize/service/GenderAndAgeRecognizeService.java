package com.csair.ai.vision.facerecognize.service;

import java.util.HashMap;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.csair.ai.tensorflow.TensorflowServer;
import com.csair.ai.vision.facerecognize.common.PropertyLoader;
import com.csair.ai.vision.facerecognize.core.classifier.AgeClassifier;
import com.csair.ai.vision.facerecognize.core.classifier.GenderClassifier;

@Service("genderAndAgeRecognizeService")
@Transactional
public class GenderAndAgeRecognizeService {

	@Autowired
	private PropertyLoader propertyLoader;
	private static Map<String, TensorflowServer> serverMap;
	
	public Map<String, String> getGenderAndAge(byte[] imgByte) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Map<String, String> res = new HashMap<String, String>();
		initTFServerMap();
		Mat img = Imgcodecs.imdecode(new MatOfByte(imgByte),
				Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
		GenderClassifier genClassifier = new GenderClassifier();
		double[] genProb = genClassifier.classify(serverMap.get("gender"), img);
		String[] genOpt = {"M", "F"};
		res.put("gender", getStrAns(genOpt, genProb, genOpt.length));
		
		AgeClassifier ageClassifier = new AgeClassifier();
		double[] ageProb = ageClassifier.classify(serverMap.get("age"), img);
//		String[] ageOpt = {"0~2", "4~6", "8~12", "15~20", "25~32", "38~43", "48~53", "60~100"};
		String[] ageOpt = {"0~2", "3~5", "6~10", "11~18", "19~27", "28~33", "34~43", "44~100"};
		res.put("age", getStrAns(ageOpt, ageProb, ageOpt.length));
		return res;
	}
	
	private String getStrAns(String[] option, double[] prob, int maxLength) {
		String ans = "";
		if(maxLength >= 1) {
			int index = 0;
			if(option != null && option.length >= maxLength
					&& prob != null && prob.length >= maxLength) {
				for(int i = 0; i < maxLength; i++) {
					if(prob[i] > prob[index]) {
						index = i;
					}
				}
				ans = option[index];
			}
		}
		return ans;
	}
	
	private void initTFServerMap() {
		if (serverMap == null || serverMap.isEmpty()) {
			serverMap = new HashMap<String, TensorflowServer>();
			serverMap.put("gender",
					new TensorflowServer(propertyLoader.getStringProperty("classifier.gender.host"),
							propertyLoader.getIntegerProperty("classifier.gender.port"),
							propertyLoader.getStringProperty("classifier.gender.model.name"),
							propertyLoader.getIntegerProperty("classifier.gender.model.v")));
			serverMap.put("age",
					new TensorflowServer(propertyLoader.getStringProperty("classifier.age.host"),
							propertyLoader.getIntegerProperty("classifier.age.port"),
							propertyLoader.getStringProperty("classifier.age.model.name"),
							propertyLoader.getIntegerProperty("classifier.age.model.v")));
		}
	}
	
}
