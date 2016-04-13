package fr.liglab.esprit.binarization.softmax;

import java.util.Comparator;

public class SoftMaxConfig implements Comparable<SoftMaxConfig> {
	public static final ConfigComparator comparator = new ConfigComparator();
	public final int bias;
	public final int nbPosWeights;
	public final int nbNegWeights;
	public double score;

	public SoftMaxConfig(int bias, int nbPosWeights, int nbNegWeights, double score) {
		super();
		this.bias = bias;
		this.nbPosWeights = nbPosWeights;
		this.nbNegWeights = nbNegWeights;
		this.score = score;
	}

	protected final int getBias() {
		return bias;
	}

	public final void setScore(double score) {
		this.score = score;
	}

	public final int getNbPosWeights() {
		return nbPosWeights;
	}

	public final int getNbNegWeights() {
		return nbNegWeights;
	}

	public final double getScore() {
		return score;
	}

	@Override
	public String toString() {
		return "SoftMaxConfig [bias=" + bias + ", nbPosWeights=" + nbPosWeights + ", nbNegWeights=" + nbNegWeights
				+ ", score=" + score + "]";
	}

	@Override
	public int compareTo(SoftMaxConfig o) {
		int c = Double.compare(this.score, o.score);
		if (c != 0) {
			return c;
		} else {
			c = this.nbPosWeights - o.nbPosWeights;
			if (c != 0) {
				return c;
			} else {
				return this.nbNegWeights - o.nbNegWeights;
			}
		}
	}

	public static class ConfigComparator implements Comparator<SoftMaxConfig> {

		@Override
		public int compare(SoftMaxConfig o1, SoftMaxConfig o2) {
			return o1.compareTo(o2);
		}

	}

}