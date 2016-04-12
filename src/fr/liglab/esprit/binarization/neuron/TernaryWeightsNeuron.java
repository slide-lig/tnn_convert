package fr.liglab.esprit.binarization.neuron;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class TernaryWeightsNeuron implements TernaryOutputNeuron {
	private final double[] weights;
	private final int th;
	private final int tl;

	public TernaryWeightsNeuron(double[] weights, int th, int tl) {
		super();
		this.weights = weights;
		this.th = th;
		this.tl = tl;
	}

	public TernaryWeightsNeuron(double[] weights, double twPos, double twNeg, int th, int tl) {
		super();
		this.weights = weights;
		this.th = th;
		this.tl = tl;
		for (int i = 0; i < weights.length; i++) {
			if (this.weights[i] > twPos) {
				this.weights[i] = 1;
			} else if (this.weights[i] < twNeg) {
				this.weights[i] = -1;
			} else {
				this.weights[i] = 0;
			}
		}
	}

	public TernaryWeightsNeuron(double[] weights, int twPosIndex, int twNegIndex, int th, int tl) {
		super();
		this.weights = weights;
		this.th = th;
		this.tl = tl;
		for (int i = 0; i < weights.length; i++) {
			if (i > twPosIndex) {
				this.weights[i] = 1;
			} else if (i < twNegIndex) {
				this.weights[i] = -1;
			} else {
				this.weights[i] = 0;
			}
		}
	}

	public int getSum(boolean[] input) {
		int sum = 0;
		for (int i = 0; i < input.length; i++) {
			if (input[i]) {
				if (this.weights[i] > 0.) {
					sum++;
				} else if (this.weights[i] < 0.) {
					sum--;
				}
			}
		}
		return sum;
	}

	public TernaryProbDistrib getOutputProbs(boolean[] input) {
		double[] probs = new double[3];
		int sum = 0;
		for (int i = 0; i < input.length; i++) {
			if (input[i]) {
				if (this.weights[i] > 0.) {
					sum++;
				} else if (this.weights[i] < 0.) {
					sum--;
				}
			}
		}
		if (sum > this.th) {
			probs[0] = 0.;
			probs[1] = 0.;
			probs[2] = 1.;
		} else if (sum < this.tl) {
			probs[0] = 1.;
			probs[1] = 0.;
			probs[2] = 0.;
		} else {
			probs[0] = 0;
			probs[1] = 1.;
			probs[2] = 0.;
		}
		return new TernaryProbDistrib(probs);
	}

	public final double[] getWeights() {
		return this.weights;
	}
}
