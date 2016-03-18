package fr.liglab.esprit.binarization;

import java.util.Arrays;

public class TernaryProbDistrib {
	private final double[] probs;

	public TernaryProbDistrib() {
		this.probs = new double[3];
		Arrays.fill(this.probs, 0.);
	}

	public TernaryProbDistrib(double[] p) {
		if (p.length != 3) {
			throw new RuntimeException("length should be 3");
		}
		this.probs = p;
	}

	public void merge(double[] outputProbs) {
		for (int i = 0; i < outputProbs.length; i++) {
			this.probs[i] += outputProbs[i];
		}
	}

	public void merge(TernaryProbDistrib outputProbs) {
		for (int i = 0; i < this.probs.length; i++) {
			this.probs[i] += outputProbs.probs[i];
		}
	}

	public final double[] getProbs() {
		return probs;
	}

	@Override
	public String toString() {
		return "TernaryProbDistrib [probs=" + Arrays.toString(probs) + "]";
	}
}