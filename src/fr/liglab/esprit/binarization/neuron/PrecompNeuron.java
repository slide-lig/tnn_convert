package fr.liglab.esprit.binarization.neuron;

import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.util.concurrent.AtomicDouble;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

//NOT THREAD SAFE
public class PrecompNeuron implements TernaryOutputNeuron {

	private final float[] activations;
	private final double[] realWeights;
	private final boolean deterministic;
	private AtomicDouble accumAgreement;
	private AtomicInteger nbSamplesProcessed;

	public PrecompNeuron(double[] realWeights, boolean deterministic, float[] activations) {
		super();
		this.realWeights = realWeights;
		this.deterministic = deterministic;
		this.accumAgreement = new AtomicDouble(0.);
		this.nbSamplesProcessed = new AtomicInteger();
		this.activations = activations;
	}

	public double getMaxAgreement() {
		return this.accumAgreement.get() / this.nbSamplesProcessed.get();
	}

	public TernaryProbDistrib getOutputProbs(byte[] input) {
		final float out = this.activations[this.nbSamplesProcessed.get()];
		this.nbSamplesProcessed.incrementAndGet();
		final double[] outArray = new double[3];
		// output index: -1->0 0->1 1->2
		if (this.deterministic) {
			this.accumAgreement.addAndGet(1.0);
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
				this.accumAgreement.addAndGet(Math.max(outArray[1], outArray[2]));
			} else if (out == 0.f) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
				this.accumAgreement.addAndGet(1.0);
			} else {
				outArray[0] = -out;
				outArray[1] = 1. + out;
				outArray[2] = 0.;
				this.accumAgreement.addAndGet(Math.max(outArray[1], outArray[0]));
			}
		}
		return new TernaryProbDistrib(outArray);
	}

	@Override
	public TernaryProbDistrib getConvOutputProbs(byte[] input, int startX, int startY, int dataXSize, int dataYSize,
			short convXSize, short convYSize, int nbChannels) {
		return this.getOutputProbs(null);
	}

	@Override
	public double[] getWeights() {
		return this.realWeights;
	}
}
