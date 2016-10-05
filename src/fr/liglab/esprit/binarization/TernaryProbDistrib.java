package fr.liglab.esprit.binarization;

public class TernaryProbDistrib {
	private double pMin1 = 0.;
	private double p0 = 0.;
	private double p1 = 0.;

	// private final double[] probs;

	public TernaryProbDistrib() {
	}

	public void merge(TernaryProbDistrib outputProbs) {
		this.pMin1 += outputProbs.pMin1;
		this.p0 += outputProbs.p0;
		this.p1 += outputProbs.p1;
	}

	public TernaryProbDistrib(double pMin1, double p0, double p1) {
		super();
		this.pMin1 = pMin1;
		this.p0 = p0;
		this.p1 = p1;
	}

	public TernaryProbDistrib(double[] pArray) {
		super();
		this.pMin1 = pArray[0];
		this.p0 = pArray[1];
		this.p1 = pArray[2];
	}

	@Override
	public String toString() {
		return "TernaryProbDistrib [pMin1=" + pMin1 + ", p0=" + p0 + ", p1=" + p1 + "]";
	}

	public final double getPMin1() {
		return pMin1;
	}

	public final double getP0() {
		return p0;
	}

	public final double getP1() {
		return p1;
	}

	public final double getProb(int index) {
		if (index == 0) {
			return this.pMin1;
		} else if (index == 1) {
			return this.p0;
		} else {
			return this.p1;
		}
	}

	// public final double[] getProbs() {
	// return probs;
	// }

}