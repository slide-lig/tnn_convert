package fr.liglab.esprit.binarization.transformer;

public class TernaryConfig {
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

	protected final int getTh() {
		return th;
	}

	protected final int getTl() {
		return tl;
	}

	protected final int getNbPosWeights() {
		return nbPosWeights;
	}

	protected final int getNbNegWeights() {
		return nbNegWeights;
	}

	protected final double getScore() {
		return score;
	}

	@Override
	public String toString() {
		return "TernaryConfig [th=" + th + ", tl=" + tl + ", nbPosWeights=" + nbPosWeights + ", nbNegWeights="
				+ nbNegWeights + ", score=" + score + "]";
	}

}