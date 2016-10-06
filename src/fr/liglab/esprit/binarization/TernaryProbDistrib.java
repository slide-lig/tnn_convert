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

	// public final double getPMin1() {
	// return pMin1;
	// }
	//
	// public final double getP0() {
	// return p0;
	// }
	//
	// public final double getP1() {
	// return p1;
	// }

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = Double.doubleToLongBits(p0);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(p1);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = Double.doubleToLongBits(pMin1);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		TernaryProbDistrib other = (TernaryProbDistrib) obj;
		if (Double.doubleToLongBits(p0) != Double.doubleToLongBits(other.p0))
			return false;
		if (Double.doubleToLongBits(p1) != Double.doubleToLongBits(other.p1))
			return false;
		if (Double.doubleToLongBits(pMin1) != Double.doubleToLongBits(other.pMin1))
			return false;
		return true;
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