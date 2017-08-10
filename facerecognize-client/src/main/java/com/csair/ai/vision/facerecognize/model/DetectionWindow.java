package com.csair.ai.vision.facerecognize.model;

public class DetectionWindow {

	private int colb;
	private int cole;
	private int rowb;
	private int rowe;

	public DetectionWindow() {
		super();
	}
	
	public DetectionWindow(int[] box) {
		if (box != null && box.length >= 4) {
			colb = box[0];
			rowb = box[1];
			cole = box[2];
			rowe = box[3];
		} else {
			colb = 0;
			rowb = 0;
			cole = 0;
			rowe = 0;
		}
	}

	public int getColb() {
		return colb;
	}

	public void setColb(int colb) {
		this.colb = colb;
	}

	public int getCole() {
		return cole;
	}

	public void setCole(int cole) {
		this.cole = cole;
	}

	public int getRowb() {
		return rowb;
	}

	public void setRowb(int rowb) {
		this.rowb = rowb;
	}

	public int getRowe() {
		return rowe;
	}

	public void setRowe(int rowe) {
		this.rowe = rowe;
	}

}
