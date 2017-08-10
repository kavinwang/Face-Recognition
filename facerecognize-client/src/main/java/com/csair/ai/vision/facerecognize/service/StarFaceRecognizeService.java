package com.csair.ai.vision.facerecognize.service;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.csair.ai.tensorflow.TensorflowServer;
import com.csair.ai.vision.facerecognize.common.PropertyLoader;
import com.csair.ai.vision.facerecognize.core.classifier.StarFaceClassifier;

@Service("starFaceRecognizeService")
@Transactional
public class StarFaceRecognizeService {

	@Autowired
	private PropertyLoader propertyLoader;
	private static Map<String, TensorflowServer> serverMap;
	private static Map<Integer, String> starNameMap;
	
	public String recognize(double[] emb) {
		String name = "";
		initTFServerMap();
		initStarFaceInfo();
		if(emb != null && emb.length > 1) {
			StarFaceClassifier classifier = new StarFaceClassifier();
			double[] predict = classifier.classify(serverMap.get("starface"), emb);
			if(predict != null && predict.length > 1) {
				int index = (int) predict[0];
				name = starNameMap.get(index);
			}
		}
		return name;
	}
	
	private void initTFServerMap() {
		if (serverMap == null || serverMap.isEmpty()) {
			serverMap = new HashMap<String, TensorflowServer>();
			serverMap.put("starface",
					new TensorflowServer(propertyLoader.getStringProperty("classifier.starface.host"),
							propertyLoader.getIntegerProperty("classifier.starface.port"),
							propertyLoader.getStringProperty("classifier.starface.model.name"),
							propertyLoader.getIntegerProperty("classifier.starface.model.v")));
		}
	}
	
	private void initStarFaceInfo() {
		if(starNameMap == null || starNameMap.isEmpty()) {
			starNameMap = new HashMap<Integer, String>();
			BufferedReader reader = null;
			try {
				reader = new BufferedReader(new InputStreamReader(new FileInputStream(
						propertyLoader.getStringProperty("starface.name.file")), "UTF-8"));
				String str = null;
				while ((str = reader.readLine()) != null) {
					String[] map = str.split(":");
					starNameMap.put(Integer.parseInt(map[0].trim()), map[1].trim());
				}
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				if (reader != null) {
					try {
						reader.close();
					} catch (IOException e1) {
						e1.printStackTrace();
					}
				}
			}
		}
	}
	
}
