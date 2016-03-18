package fr.liglab.esprit.binarization.neuron;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public interface TernaryOutputNeuron {
	public TernaryProbDistrib getOutputProbs(boolean[] input);

	public double[] getWeights();

	default public double getWeightSign(int index) {
		return Math.signum(this.getWeights()[index]);
	}
}
