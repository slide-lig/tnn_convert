package fr.liglab.esprit.binarization.neuron;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public interface TernaryOutputNeuron {
	public TernaryProbDistrib getOutputProbs(byte[] input);

	public TernaryProbDistrib getConvOutputProbs(byte[] input, int startX, int startY, int dataXSize, short convXSize,
			short convYSize);

	public double[] getWeights();

	default public double getWeightSign(int index) {
		return Math.signum(this.getWeights()[index]);
	}

	default public int getNbPosWeights() {
		int count = 0;
		for (double d : this.getWeights()) {
			if (d > 0.) {
				count++;
			}
		}
		return count;
	}

	default public int getNbNegWeights() {
		int count = 0;
		for (double d : this.getWeights()) {
			if (d < 0.) {
				count++;
			}
		}
		return count;
	}
}
