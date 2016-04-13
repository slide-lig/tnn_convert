package fr.liglab.esprit.binarization.transformer;

import java.util.Comparator;

public class TernaryConfig implements Comparable<TernaryConfig> {
	public static final ConfigComparator comparator = new ConfigComparator();
	public final int th;
	public final int tl;
	public final int nbPosWeights;
	public final int nbNegWeights;
	public final double score;

	public TernaryConfig(int th, int tl, int nbPosWeights, int nbNegWeights, double score) {
		super();
		this.th = th;
		this.tl = tl;
		this.nbPosWeights = nbPosWeights;
		this.nbNegWeights = nbNegWeights;
		this.score = score;
	}

	public final int getTh() {
		return th;
	}

	public final int getTl() {
		return tl;
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
		return "TernaryConfig [th=" + th + ", tl=" + tl + ", nbPosWeights=" + nbPosWeights + ", nbNegWeights="
				+ nbNegWeights + ", score=" + score + "]";
	}

	@Override
	public int compareTo(TernaryConfig o) {
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

	public static class ConfigComparator implements Comparator<TernaryConfig> {

		@Override
		public int compare(TernaryConfig o1, TernaryConfig o2) {
			return o1.compareTo(o2);
		}

	}

}