package fr.liglab.esprit.binarization.neuron;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class TanHNeuron implements TernaryOutputNeuron {
	private final double[] realWeights;
	private final double bias;
	private final boolean deterministic;

	public TanHNeuron(double[] realWeights, double bias, boolean deterministic) {
		super();
		this.realWeights = realWeights;
		this.bias = bias;
		this.deterministic = deterministic;
	}

	public TernaryProbDistrib getOutputProbs(boolean[] input) {
		double[] outArray = new double[3];
		double sum = bias;
		for (int i = 0; i < input.length; i++) {
			if (input[i]) {
				sum += this.realWeights[i];
			}
		}
		double out = Math.tanh(sum);
		// output index: -1->0 0->1 1->2
		if (this.deterministic) {
			if (out > 0) {
				outArray[0] = 0.;
				outArray[1] = 0.;
				outArray[2] = 1.;
			} else if (out == 0) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
			} else {
				outArray[0] = 1;
				outArray[1] = 0.;
				outArray[2] = 0.;
			}
		} else {
			if (out > 0) {
				outArray[0] = 0.;
				outArray[1] = 1. - out;
				outArray[2] = out;
			} else if (out == 0) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
			} else {
				outArray[0] = -out;
				outArray[1] = 1. + out;
				outArray[2] = 0.;
			}
		}
		return new TernaryProbDistrib(outArray);
	}

	public final double[] getWeights() {
		return realWeights;
	}
}
