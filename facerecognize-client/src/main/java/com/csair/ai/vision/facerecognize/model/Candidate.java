package com.csair.ai.vision.facerecognize.model;

public class Candidate {

	private String id;
	private String name;
	private double[] embeddings;
	private String image;
	private double probability;
	private String gender;
	private String ageRange;
	private String starFace;
	
	public Candidate(String id, String name) {
		this.id = id;
		this.name = name;
	}
	
	public Candidate(String id, String name, double prob) {
		this.id = id;
		this.name = name;
		this.probability = prob;
	}
	
	public Candidate(String id, String name, double[] emb) {
		this.id = id;
		this.name = name;
		this.embeddings = emb;
	}
	
	public Candidate(String id, String name, String img) {
		this.id = id;
		this.name = name;
		this.image = img;
	}
	
	public Candidate(String id, String name, double prob,
			String gender, String age, String starface) {
		this.id = id;
		this.name = name;
		this.probability = prob;
		this.gender = gender;
		this.ageRange = age;
		this.starFace = starface;
	}
	
	public Candidate(String id, String name, String img, double prob,
			String gender, String age, String starface) {
		this.id = id;
		this.name = name;
		this.image = img;
		this.probability = prob;
		this.gender = gender;
		this.ageRange = age;
		this.starFace = starface;
	}
	
	public String getId() {
		return id;
	}
	
	public void setId(String id) {
		this.id = id;
	}
	
	public String getName() {
		return name;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public double[] getEmbeddings() {
		return embeddings;
	}
	
	public void setEmbeddings(double[] embeddings) {
		this.embeddings = embeddings;
	}
	
	public String getImage() {
		return image;
	}

	public void setImage(String image) {
		this.image = image;
	}

	public double getProbability() {
		return probability;
	}

	public void setProbability(double probability) {
		this.probability = probability;
	}
	
	public String getGender() {
		return gender;
	}

	public void setGender(String gender) {
		this.gender = gender;
	}

	public String getAgeRange() {
		return ageRange;
	}

	public void setAgeRange(String ageRange) {
		this.ageRange = ageRange;
	}

	public String getStarFace() {
		return starFace;
	}

	public void setStarFace(String starFace) {
		this.starFace = starFace;
	}

	public String toMetadataString() {
		StringBuilder sb = new StringBuilder("");
		sb.append(id);
		sb.append(",");
		sb.append(name);
		sb.append(",");
		for(double emb : embeddings) {
			sb.append(emb);
			sb.append(",");
		}
		sb.deleteCharAt(sb.length() - 1);
		return sb.toString();
	}
	
	public double distance(double[] emb) {
		double dis = 0;
		if(this.embeddings != null && emb != null && this.embeddings.length == emb.length) {
			for(int i = 0; i < this.embeddings.length; i++) {
				dis += Math.pow(this.embeddings[i] - emb[i], 2);
			}
			dis = Math.sqrt(dis);
		}
		return dis;
	}
	
}
