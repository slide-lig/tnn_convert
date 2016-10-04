package fr.liglab.esprit.binarization.neuron;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class PrecompNeuron implements TernaryOutputNeuron {
	private final double[] realWeights;
	private final boolean deterministic;
	private final float[] activations;
	private double accumAgreement;
	private int nbSamplesProcessed;

	public PrecompNeuron(double[] realWeights, boolean deterministic, float[] activations) {
		super();
		this.realWeights = realWeights;
		this.deterministic = deterministic;
		this.accumAgreement = 0.;
		this.nbSamplesProcessed = 0;
		this.activations = activations;
	}

	public double getMaxAgreement() {
		return this.accumAgreement / this.nbSamplesProcessed;
	}

	public TernaryProbDistrib getOutputProbs(byte[] input) {
		final float out = this.activations[this.nbSamplesProcessed];
		this.nbSamplesProcessed++;
		final double[] outArray = new double[3];
		// output index: -1->0 0->1 1->2
		if (this.deterministic) {
			this.accumAgreement += 1.0;
			if (out > 0.f) {
				outArray[0] = 0.;
				outArray[1] = 0.;
				outArray[2] = 1.;
			} else if (out == 0.f) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
			} else {
				outArray[0] = 1;
				outArray[1] = 0.;
				outArray[2] = 0.;
			}
		} else {
			if (out > 0.f) {
				outArray[0] = 0.;
				outArray[1] = 1. - out;
				outArray[2] = out;
				this.accumAgreement += Math.max(outArray[1], outArray[2]);
			} else if (out == 0.f) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
				this.accumAgreement += 1.0;
			} else {
				outArray[0] = -out;
				outArray[1] = 1. + out;
				outArray[2] = 0.;
				this.accumAgreement += Math.max(outArray[1], outArray[0]);
			}
		}
		return new TernaryProbDistrib(outArray);
	}

	@Override
	public TernaryProbDistrib getConvOutputProbs(byte[] input, int startX, int startY, int dataXSize, short convXSize,
			short convYSize) {
		return this.getOutputProbs(null);
	}

	@Override
	public final double[] getWeights() {
		return realWeights;
	}
}
