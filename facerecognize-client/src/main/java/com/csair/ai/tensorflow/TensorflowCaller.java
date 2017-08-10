package com.csair.ai.tensorflow;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import org.tensorflow.framework.TensorShapeProto.Dim;

import com.csair.ai.tensorflow.TensorflowClient;
import com.csair.ai.tensorflow.TensorflowServer;

public class TensorflowCaller {

	private Map<String, TensorProto> call(TensorflowServer server, String key, double[][][][] value) {
		Map<String, TensorProto> response = new HashMap<String, TensorProto>();
		TensorflowClient client = new TensorflowClient(server.getHost(), server.getPort());
		try {
			response = client.inference(server.getModelName(), server.getModelVersion(), buildInput(key, value));
		} catch(Exception e) {
			e.printStackTrace();
		} finally {
			try {
				client.shutdown();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		return response;
	}

	public Map<String, TensorProto> buildInput(String key, double[][] element) {
		int dim1L = element.length, dim2L = element[0].length;
		TensorProto.Builder builder = TensorProto.newBuilder();
		for (int d1 = 0; d1 < dim1L; d1++) {
			for (int d2 = 0; d2 < dim2L; d2++) {
				builder.addFloatVal((float) element[d1][d2]);
			}
		}
		
		TensorShapeProto.Dim dim1 = TensorShapeProto.Dim.newBuilder().setSize(dim1L).build(),
				dim2 = TensorShapeProto.Dim.newBuilder().setSize(dim2L).build();
		
		builder.setDtype(DataType.DT_FLOAT).setTensorShape(
				TensorShapeProto.newBuilder().addDim(dim1).addDim(dim2).build());
		
		Map<String, TensorProto> input = new HashMap<String, TensorProto>();
		input.put(key, builder.build());
		return input;
	}
	
	public Map<String, TensorProto> buildInput(String key, double[][][][] element) {
		int dim1L = element.length, dim2L = element[0].length, dim3L = element[0][0].length, dim4L = element[0][0][0].length;
		TensorProto.Builder builder = TensorProto.newBuilder();
		for (int d1 = 0; d1 < dim1L; d1++) {
			for (int d2 = 0; d2 < dim2L; d2++) {
				for (int d3 = 0; d3 < dim3L; d3++) {
					for (int d4 = 0; d4 < dim4L; d4++) {
						builder.addFloatVal((float) element[d1][d2][d3][d4]);
					}
				}
			}
		}

		TensorShapeProto.Dim dim1 = TensorShapeProto.Dim.newBuilder().setSize(dim1L).build(),
				dim2 = TensorShapeProto.Dim.newBuilder().setSize(dim2L).build(),
				dim3 = TensorShapeProto.Dim.newBuilder().setSize(dim3L).build(),
				dim4 = TensorShapeProto.Dim.newBuilder().setSize(dim4L).build();

		builder.setDtype(DataType.DT_FLOAT).setTensorShape(
				TensorShapeProto.newBuilder().addDim(dim1).addDim(dim2).addDim(dim3).addDim(dim4).build());

		Map<String, TensorProto> input = new HashMap<String, TensorProto>();
		input.put(key, builder.build());
		return input;
	}
	
	public Map<String, double[][]> get2DValue(TensorflowServer server, String key, double[][][][] value) {
		Map<String, TensorProto> response = call(server, key, value);
		Map<String, double[][]> result = new HashMap<String, double[][]>();
		for (Map.Entry<String, TensorProto> entry : response.entrySet()) {
			result.put(entry.getKey(), parseResponseD2(entry.getValue()));
		}
		return result;
	}
	
	public Map<String, double[][][][]> get4DValue(TensorflowServer server, String key, double[][][][] value) {
		Map<String, TensorProto> response = call(server, key, value);
		Map<String, double[][][][]> result = new HashMap<String, double[][][][]>();
		for (Map.Entry<String, TensorProto> entry : response.entrySet()) {
			result.put(entry.getKey(), parseResponseD4(entry.getValue()));
		}
		return result;
	}
	
	public double[][] parseResponseD2(TensorProto tensor) {
		List<Dim> dims = tensor.getTensorShape().getDimList();
		int dim1 = (int) dims.get(0).getSize(), dim2 = (int) dims.get(1).getSize();
		double[][] res = new double[dim1][dim2];
		int index = 0;
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++) {
				res[i][j] = tensor.getFloatVal(index++);
			}
		}
		return res;
	}
	
	public double[][][][] parseResponseD4(TensorProto tensor) {
		List<Dim> dims = tensor.getTensorShape().getDimList();
		int dim1 = (int) dims.get(0).getSize(), dim2 = (int) dims.get(1).getSize(), dim3 = (int) dims.get(2).getSize(),
				dim4 = (int) dims.get(3).getSize();
		double[][][][] res = new double[dim1][dim2][dim3][dim4];
		int index = 0;
		for (int i = 0; i < dim1; i++) {
			for (int j = 0; j < dim2; j++) {
				for (int k = 0; k < dim3; k++) {
					for (int l = 0; l < dim4; l++) {
						res[i][j][k][l] = tensor.getFloatVal(index++);
					}
				}
			}
		}
		return res;
	}

	public long[] parseResponseL1(TensorProto tensor) {
		List<Dim> dims = tensor.getTensorShape().getDimList();
		int dim1 = (int) dims.get(0).getSize();
		long[] res = new long[dim1];
		int index = 0;
		for (int i = 0; i < dim1; i++) {
			res[i] = tensor.getInt64Val(index++);
		}
		return res;
	}
	
}
