package com.csair.ai.vision.facerecognize.common;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.core.env.Environment;

@Configuration
@PropertySource({ "classpath:app.properties" })
public class PropertyLoader {

	@Autowired
	private Environment env;

	public String getStringProperty(String key) {
		return env.getProperty(key);
	}
	
	public Integer getIntegerProperty(String key) {
		return Integer.parseInt(env.getProperty(key));
	}

}
