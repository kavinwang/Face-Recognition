package com.csair.ai.vision.facerecognize.service;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
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
import com.csair.ai.vision.facerecognize.common.PropertyLoader;
import com.csair.ai.vision.facerecognize.core.classifier.CsairEmployeeClassifier;
import com.csair.ai.vision.facerecognize.core.recognize.FaceNet;
import com.csair.ai.vision.facerecognize.model.Candidate;

@Service("csairEmployeeRecognizeService")
@Transactional
public class CsairEmployeeRecognizeService {

	@Autowired
	private PropertyLoader propertyLoader;
	private static Map<String, TensorflowServer> serverMap;
	private static Map<Integer, String> candidateIndexMap;
	private static Map<String, String> candidateNameMap;
	@Autowired
	private GenderAndAgeRecognizeService genderAndAgeRecognizeService;
	@Autowired
	private StarFaceRecognizeService starFaceRecognizeService;

	public Candidate recognize(byte[] imgByte) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		initCandidateInfo();
		initTFServerMap();
		Mat img = Imgcodecs.imdecode(new MatOfByte(imgByte),
				Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
		FaceNet facenet = new FaceNet();
		
		Candidate candidate = null;
		double[] emb = facenet.calcEmbeddings(serverMap.get("facenet"), img);
		if (emb != null && emb.length > 1) {
			CsairEmployeeClassifier classifier = new CsairEmployeeClassifier();
			double[] res = classifier.classify(serverMap.get("classifier"), emb);
			String emno = candidateIndexMap.get((int)res[0]);
			candidate = new Candidate(emno, candidateNameMap.get(emno), res[1]);
		}
		return candidate;
	}
	
	public Candidate recognizeAll(byte[] imgByte) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		initCandidateInfo();
		initTFServerMap();
		Mat img = Imgcodecs.imdecode(new MatOfByte(imgByte),
				Imgcodecs.CV_LOAD_IMAGE_UNCHANGED);
		FaceNet facenet = new FaceNet();
		Candidate candidate = null;
		double[] emb = facenet.calcEmbeddings(serverMap.get("facenet"), img);
		if (emb != null && emb.length > 1) {
			CsairEmployeeClassifier classifier = new CsairEmployeeClassifier();
			double[] res = classifier.classify(serverMap.get("classifier"), emb);
			String emno = candidateIndexMap.get((int)res[0]), gender = "", age = "";
			String starface = starFaceRecognizeService.recognize(emb);
			Map<String, String> genAndAge = genderAndAgeRecognizeService.getGenderAndAge(imgByte);
			if(genAndAge != null && genAndAge.containsKey("gender") && genAndAge.containsKey("age")) {
				gender = genAndAge.get("gender");
				age = genAndAge.get("age");
			}
			candidate = new Candidate(emno, candidateNameMap.get(emno), res[1],
					gender, age, starface);
		}
		return candidate;
	}
	
	public void generateCandidateMetadata(String metadataFile) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		initCandidateInfo();
		initTFServerMap();
		File folder = new File(propertyLoader.getStringProperty("candidate.metadata.folder"));
		Map<String, List<String>> classMap = new HashMap<String, List<String>>();
		if (folder != null && folder.isDirectory()) {
			File[] classList = folder.listFiles();
			for (File cl : classList) {
				if (cl.isDirectory()) {
					File[] eles = cl.listFiles();
					List<String> paths = new ArrayList<String>();
					for (File ele : eles) {
						paths.add(ele.getAbsolutePath());
					}
					classMap.put(cl.getName(), paths);
				}
			}
		}

		FaceNet facenet = new FaceNet();
		List<Candidate> candidates = new ArrayList<Candidate>();
		List<String> failed = new ArrayList<String>();
		for (Map.Entry<String, List<String>> entry : classMap.entrySet()) {
			String candidate = entry.getKey();
			List<String> images = entry.getValue();
			for (String imagePath : images) {
				Mat img = Imgcodecs.imread(imagePath);
				double[] emb = facenet.calcEmbeddings(serverMap.get("facenet"), img);
				if (emb != null && emb.length > 1) {
					System.out.println("processing: " + imagePath + ", OK");
					candidates.add(new Candidate(candidate, candidateNameMap.get(candidate), emb));
				} else {
					failed.add(imagePath);
					System.out.println("processing: " + imagePath + ", FAILED");
				}
			}
		}

		try {
			FileWriter writer = new FileWriter(metadataFile);
			for (Candidate cand : candidates) {
				writer.write(cand.toMetadataString() + "\n");
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void initTFServerMap() {
		if (serverMap == null || serverMap.isEmpty()) {
			serverMap = new HashMap<String, TensorflowServer>();
			serverMap.put("facenet",
					new TensorflowServer(propertyLoader.getStringProperty("facenet.host"),
							propertyLoader.getIntegerProperty("facenet.port"),
							propertyLoader.getStringProperty("facenet.model.name"),
							propertyLoader.getIntegerProperty("facenet.model.v")));
			serverMap.put("classifier",
					new TensorflowServer(propertyLoader.getStringProperty("classifier.host"),
							propertyLoader.getIntegerProperty("classifier.port"),
							propertyLoader.getStringProperty("classifier.model.name"),
							propertyLoader.getIntegerProperty("classifier.model.v")));
		}
	}

	private void initCandidateInfo() {
		if(candidateNameMap == null || candidateNameMap.isEmpty()) {
			candidateNameMap = new HashMap<String, String>();
			BufferedReader reader = null;
			try {
				reader = new BufferedReader(new InputStreamReader(new FileInputStream(
						propertyLoader.getStringProperty("candidate.name.file")), "UTF-8"));
				String str = null;
				while ((str = reader.readLine()) != null) {
					String[] map = str.split(":");
					candidateNameMap.put(map[0].trim(), map[1].trim());
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
		
		if(candidateIndexMap == null || candidateIndexMap.isEmpty()) {
			candidateIndexMap = new HashMap<Integer, String>();
			File folder = new File(propertyLoader.getStringProperty("candidate.folder"));
			if(folder != null && folder.isDirectory()) {
				File[] files = folder.listFiles();
				if(files != null && files.length > 0) {
					for(int i = 0; i < files.length; i++) {
						candidateIndexMap.put(i, files[i].getName());
					}
				}
			}
		}
	}

}
