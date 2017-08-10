package com.csair.ai.vision.facerecognize.controller;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.codec.binary.Base64;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import com.csair.ai.vision.facerecognize.model.Candidate;
import com.csair.ai.vision.facerecognize.model.DetectionWindow;
import com.csair.ai.vision.facerecognize.service.CsairEmployeeRecognizeService;
import com.csair.ai.vision.facerecognize.service.FaceDetectService;

@RestController
@RequestMapping("/service")
public class ServiceController {

	@Autowired
	private FaceDetectService faceDetectService;
	@Autowired
	private CsairEmployeeRecognizeService csairEmployeeRecognizeService;

	@RequestMapping(value = "/detectFaces/{mode}", method = RequestMethod.POST)
	public List<DetectionWindow> detectFaces(@PathVariable("mode") int mode, @RequestBody String data) {
		List<DetectionWindow> windows = new ArrayList<DetectionWindow>();
		if(mode < 0 || mode > 1) {
			mode = 0;
		}
		if (data != null && !"".equals(data.trim())) {
			byte[] image = Base64.decodeBase64(data.trim());
			windows = faceDetectService.mtcnn(image, mode);
		}
		return windows;
	}

	@RequestMapping(value = "/recognizeFaces", method = RequestMethod.POST)
	public Candidate recognizeFaces(@RequestBody String data) {
		if (data != null && !"".equals(data.trim())) {
			byte[] image = Base64.decodeBase64(data.trim());
			return csairEmployeeRecognizeService.recognizeAll(image);
		} else {
			return null;
		}
	}
	
	@RequestMapping(value = "/generateMetadata", method = RequestMethod.POST)
	public void generateMetadata(@RequestBody String metadataFile) {
		csairEmployeeRecognizeService.generateCandidateMetadata(metadataFile);
	}
	
}
