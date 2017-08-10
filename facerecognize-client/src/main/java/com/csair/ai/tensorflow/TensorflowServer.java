package com.csair.ai.tensorflow;

public class TensorflowServer {

	private String Host;
	private int Port;
	private String ModelName;
	private long ModelVersion;

	public TensorflowServer(String host, int port, String modelName, long modelVersion) {
		this.Host = host;
		this.Port = port;
		this.ModelName = modelName;
		this.ModelVersion = modelVersion;
	}

	public String getHost() {
		return Host;
	}

	public int getPort() {
		return Port;
	}

	public String getModelName() {
		return ModelName;
	}

	public long getModelVersion() {
		return ModelVersion;
	}

}
