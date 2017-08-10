package com.csair.ai.vision.facerecognize.client;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.apache.commons.codec.binary.Base64;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.springframework.web.client.RestTemplate;

import com.csair.ai.tensorflow.TensorflowServer;
import com.csair.ai.vision.facerecognize.core.classifier.GenderClassifier;
import com.csair.ai.vision.facerecognize.core.detect.MTCNN;
import com.csair.ai.vision.facerecognize.core.recognize.FaceNet;
import com.csair.ai.vision.facerecognize.model.Candidate;
import com.csair.ai.vision.facerecognize.model.DetectionWindow;

public class FaceRecognizeClient {

	@Test
	public void testDetectFaces() {
		System.out.println("Testing detect faces API----------");
		String REST_SERVICE_URI = "http://localhost:8080/facerecognize/service/";
//		String img = "/home/zero/1_workspace/CSAIR/CSAIR_FaceRecognize/original_photos/196717/IMG_0181.JPG";
		String img = "C:/Users/zero/Downloads/kb1.png";
		InputStream in = null;
		byte[] data = null;
		try {
			in = new FileInputStream(img);
			data = new byte[in.available()];
			in.read(data);
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		String dataStr = new String(Base64.encodeBase64(data));
		RestTemplate restTemplate = new RestTemplate();
		String item = restTemplate.postForObject(REST_SERVICE_URI + "detect/png", dataStr, String.class);
		System.out.println(item);
		
		if(item != null) {
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			Mat src = Imgcodecs.imread(img);
			String[] items = item.split(";");
			if(items.length > 0) {
				for(String tmp : items) {
					String[] tmpArr = tmp.split(",");
					double[] points = new double[4];
					points[0] = Double.parseDouble(tmpArr[0]);
					points[1] = Double.parseDouble(tmpArr[1]);
					points[2] = Double.parseDouble(tmpArr[2]);
					points[3] = Double.parseDouble(tmpArr[3]);
//					Core.rectangle(src, new Point(points[0], points[1]), new Point(points[2], points[3]), new Scalar(255, 255, 255));
				}
			}
			Imgcodecs.imwrite("C:/Users/zero/Downloads/test.png", src);
		}
	}
	
	@Test
	public void testRecognize() {
		System.out.println("Testing recognize API----------");
		String REST_SERVICE_URI = "http://localhost:8080/facerecognize/service/";
		String img = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails/761809/0b7b3fb0-47b8-4df1-8a82-d7fd95440690.png";
		InputStream in = null;
		byte[] data = null;
		try {
			in = new FileInputStream(img);
			data = new byte[in.available()];
			in.read(data);
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		String dataStr = new String(Base64.encodeBase64(data));
		RestTemplate restTemplate = new RestTemplate();
		String item = restTemplate.postForObject(REST_SERVICE_URI + "recognizeFaces/", dataStr, String.class);
		System.out.println(item);
	}

	@Test
	public void testGenerateCandidateMetadata() {
		System.out.println("Testing generate candidate metadata API----------");
		String REST_SERVICE_URI = "http://localhost:8080/facerecognize/service/";
		String metadataFile = "E:/2_workdatas/csair/facerecognize/release/candidate/candidate_metadata";
		RestTemplate restTemplate = new RestTemplate();
		restTemplate.postForObject(REST_SERVICE_URI + "generateMetadata", metadataFile, String.class);
		System.out.println("Complete generate candidate metadata API----------");
	}

	@Test
	public void testAccuracy() {
		String REST_SERVICE_URI = "http://localhost:8080/facerecognize/face/";
		File folder = new File("/home/zero/1_workspace/CSAIR/CSAIR_FaceRecognize/original_photos");
		int total = 0, correct = 0, error = 0, failed = 0;
		List<String> errorList = new ArrayList<String>(), failedList = new ArrayList<String>();
		long startTime = System.currentTimeMillis();
		if (folder != null && folder.isDirectory()) {
			File[] classList = folder.listFiles();
			for (File cl : classList) {
				if (cl.isDirectory()) {
					File[] eles = cl.listFiles();
					List<String> paths = new ArrayList<String>();
					for (File ele : eles) {
						System.out.println("Processing " + ele.getAbsolutePath());
						total++;
						InputStream in = null;
						byte[] data = null;
						try {
							in = new FileInputStream(ele.getAbsolutePath());
							data = new byte[in.available()];
							in.read(data);
							in.close();
						} catch (IOException e) {
							e.printStackTrace();
						}
						String dataStr = new String(Base64.encodeBase64(data));
						RestTemplate restTemplate = new RestTemplate();
						String item = restTemplate.postForObject(REST_SERVICE_URI + "recognize/1/jpg", dataStr,
								String.class);
						if (item == null || item.equals("")) {
							failed++;
							failedList.add(ele.getAbsolutePath());
						} else {
							String[] tmp = item.split(":");
							if (tmp[0].equals(cl.getName())) {
								correct++;
							} else {
								error++;
								errorList.add(ele.getAbsolutePath());
							}
						}
					}
				}
			}
		}
		long endTime = System.currentTimeMillis();
		double accuracy = ((double) correct * 100) / total, errorRate = ((double) error * 100) / total,
				failedRate = ((double) failed * 100) / total;

		System.out.println("accuracy: " + accuracy + "%, error: " + errorRate + "%, failed: " + failedRate);
		System.out.println("error images:");
		for (String err : errorList) {
			System.out.println(err);
		}
		System.out.println("failed images:");
		for (String fa : failedList) {
			System.out.println(fa);
		}
		System.out.println("take time: " + (endTime - startTime) + "ms, average: "
				+ ((double) (endTime - startTime)) / total + "ms per image");
	}
	
	@Test
	public void testGenerateEmbeddings() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		File folder = new File("D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails");
		String metadataFile = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/candidate_metadata";
		List<String> datas = new ArrayList<String>();
		FaceNet facenet = new FaceNet();
		TensorflowServer server = new TensorflowServer("10.95.68.103", 9004, "facenet/inception_resnet_v1", 6);
		int cla = 0;
		
		if (folder != null && folder.isDirectory()) {
			File[] classList = folder.listFiles();
			for (File cl : classList) {
				if (cl.isDirectory()) {
					File[] eles = cl.listFiles();
					for(File ele :eles) {
						System.out.println("Processing " + ele.getAbsolutePath());
						Mat image = Imgcodecs.imread(ele.getAbsolutePath());
						double[] embs = facenet.calcEmbeddings(server, image);
						String line = cla + ",";
						for(double e : embs) {
							line += e + ",";
						}
						line = line.substring(0, line.length() - 1);
						datas.add(line);
					}
				}
				cla++;
			}
		}
		
		try {
			FileWriter writer = new FileWriter(metadataFile);
			for (String d : datas) {
				writer.write(d + "\n");
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Generate embeddings success");
	}
	
	@Test
	public void testGenerateStartEmbeddings() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		File folder = new File("D:/WorkDatas/CSAIR_FaceRecognize/v0.2/star_photos");
		String metadataFile = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/start_metadata";
		List<String> datas = new ArrayList<String>();
		FaceNet facenet = new FaceNet();
		TensorflowServer server = new TensorflowServer("10.95.68.103", 9004, "facenet/inception_resnet_v1", 4);
		
		if (folder != null && folder.isDirectory()) {
			File[] classList = folder.listFiles();
			for (File cl : classList) {
				if (cl.isDirectory()) {
					File[] eles = cl.listFiles();
					for(File ele :eles) {
						System.out.println("Processing " + ele.getAbsolutePath());
						Mat image = Imgcodecs.imread(ele.getAbsolutePath());
						double[] embs = facenet.calcEmbeddings(server, image);
						String line = cl.getName() + ",";
						for(double e : embs) {
							line += e + ",";
						}
						line = line.substring(0, line.length() - 1);
						datas.add(line);
					}
				}
			}
		}
		
		try {
			FileWriter writer = new FileWriter(metadataFile);
			for (String d : datas) {
				writer.write(d + "\n");
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Generate embeddings success");
	}
	
	@Test
	public void testTakeFaces() {
		System.out.println("Testing take face API----------");
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		File folder = new File("D:/WorkDatas/CSAIR_FaceRecognize/v0.2/star_photos_44/");
		String tarFolder = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/star_photos/";
		Map<String, TensorflowServer> serverMap = new HashMap<String, TensorflowServer>();
		serverMap.put("pnet", new TensorflowServer("10.95.68.103", 9001, "mtcnn/pnet", 1));
		serverMap.put("rnet", new TensorflowServer("10.95.68.103", 9002, "mtcnn/rnet", 1));
		serverMap.put("onet", new TensorflowServer("10.95.68.103", 9003, "mtcnn/onet", 1));
		MTCNN mtcnn = new MTCNN();
		
		if (folder != null && folder.isDirectory()) {
			File[] classList = folder.listFiles();
			for (File cl : classList) {
				if (cl.isDirectory()) {
					File tf = new File(tarFolder + cl.getName());
					tf.mkdirs();
					File[] eles = cl.listFiles();
					for(File ele : eles) {
						System.out.println("Processing " + ele.getAbsolutePath());
						Mat image = Imgcodecs.imread(ele.getAbsolutePath());
						int[][] totalBoxes = mtcnn.detectFace(serverMap, image);
						if(totalBoxes != null && totalBoxes.length > 0) {
							for(int[] box : totalBoxes) {
								Mat crop = image.rowRange(box[1], box[3]).colRange(box[0], box[2]);
								String path = tarFolder + cl.getName() + "/" + UUID.randomUUID().toString() + ".png";
								Imgcodecs.imwrite(path, crop);
							}
						}
					}
				}
			}
		}
	}
	
	@Test
	public void testGenderClassifier() {
		System.out.println("Testing gender classifier API----------");
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/180200/915779d5-89d8-4b82-b20a-a3af6f19846b.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/181231/dcf9b94e-092b-4ab3-b3cb-9a5d964e7c66.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/209140/9ff806a6-da6a-4072-83e7-d3e0f4eb2452.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/209403/0d16eac6-ca26-452b-859f-6ea79bf8d5ec.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/749662/1ec99476-80e9-4385-af05-81604e77ba75.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/750992/0aafd1dd-ab1a-40da-8c30-d3fac3fba88a.png";
		TensorflowServer server = new TensorflowServer("10.95.68.103", 9006, "gender_classifier", 1);
		Mat img = Imgcodecs.imread(imgPath);
		GenderClassifier classifier = new GenderClassifier();
		double[] prob = classifier.classify(server, img);
		for(double p : prob) {
			System.out.println(p);
		}
	}
	
	@Test
	public void testGenderAndAgeClassifier() {
		System.out.println("Testing age classifier API----------");
		String REST_SERVICE_URI = "http://localhost:8080/facerecognize/service/getGenderAndAge";
		String imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/180200/3706e4cd-76b3-4d18-8435-437e9b65d928.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/181213/5561d144-bc87-49b3-ad53-708c0a987e44.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/181231/e94f261c-6784-43ec-9eb9-f8fb321882e9.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/209140/9ff806a6-da6a-4072-83e7-d3e0f4eb2452.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/209403/0d16eac6-ca26-452b-859f-6ea79bf8d5ec.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/749662/1ec99476-80e9-4385-af05-81604e77ba75.png";
		imgPath = "D:/WorkDatas/CSAIR_FaceRecognize/v0.2/thumbnails_44/750992/0aafd1dd-ab1a-40da-8c30-d3fac3fba88a.png";
		InputStream in = null;
		byte[] data = null;
		try {
			in = new FileInputStream(imgPath);
			data = new byte[in.available()];
			in.read(data);
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		String dataStr = new String(Base64.encodeBase64(data));
		RestTemplate restTemplate = new RestTemplate();
		String item = restTemplate.postForObject(REST_SERVICE_URI, dataStr, String.class);
		System.out.println(item);
	}
	
}
