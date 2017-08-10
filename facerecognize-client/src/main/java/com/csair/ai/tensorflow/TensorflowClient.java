package com.csair.ai.tensorflow;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.tensorflow.framework.TensorProto;

import com.google.protobuf.Int64Value;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

public class TensorflowClient {

	private final int AWAIT_SECONDS = 5;
	private final ManagedChannel channel;
	private final PredictionServiceGrpc.PredictionServiceBlockingStub blockingStub;

	public TensorflowClient(String host, int port) {
		channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext(true).build();
		blockingStub = PredictionServiceGrpc.newBlockingStub(channel);
	}

	public Map<String, TensorProto> inference(String modelName, long modelVersion, Map<String, TensorProto> inputs) {
		Int64Value version = Int64Value.newBuilder().setValue(modelVersion).build();
		Model.ModelSpec modelSpec = Model.ModelSpec.newBuilder().setName(modelName).setVersion(version).build();
		Predict.PredictRequest request = Predict.PredictRequest.newBuilder().setModelSpec(modelSpec)
				.putAllInputs(inputs).build();
		Predict.PredictResponse response = blockingStub.predict(request);
		return response.getOutputsMap();
	}

	public void shutdown() throws InterruptedException {
		channel.shutdown().awaitTermination(AWAIT_SECONDS, TimeUnit.SECONDS);
	}

}
