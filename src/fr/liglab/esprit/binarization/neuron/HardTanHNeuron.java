package fr.liglab.esprit.binarization.neuron;

import java.util.concurrent.atomic.AtomicInteger;

import com.google.common.util.concurrent.AtomicDouble;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class HardTanHNeuron implements TernaryOutputNeuron {
	private final double[] realWeights;
	private final double bias;
	private final boolean deterministic;
	private AtomicDouble accumAgreement;
	private AtomicInteger nbSamplesProcessed;

	public HardTanHNeuron(double[] realWeights, double bias, boolean deterministic) {
		super();
		this.realWeights = realWeights;
		this.bias = bias;
		this.deterministic = deterministic;
		this.accumAgreement = new AtomicDouble(0.);
		this.nbSamplesProcessed = new AtomicInteger(0);
	}

	public double getMaxAgreement() {
		return this.accumAgreement.get() / this.nbSamplesProcessed.get();
	}

	public TernaryProbDistrib getOutputProbs(byte[] input) {
		this.nbSamplesProcessed.incrementAndGet();
		double[] outArray = new double[3];
		double sum = bias;
		for (int i = 0; i < input.length; i++) {
			sum += this.realWeights[i] * input[i];
		}
		double out = Math.min(Math.max(sum, -1), 1);
		if (this.deterministic) {
			this.accumAgreement.addAndGet(1.0);
			if (out > 0.) {
				outArray[0] = 0.;
				outArray[1] = 0.;
				outArray[2] = 1.;
			} else if (out == 0.) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
			} else {
				outArray[0] = 1.;
				outArray[1] = 0.;
				outArray[2] = 0.;
			}
		} else {
			if (out > 0.) {
				outArray[0] = 0.;
				outArray[1] = 1. - out;
				outArray[2] = out;
				this.accumAgreement.addAndGet(Math.max(outArray[1], outArray[2]));
			} else if (out == 0.) {
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
		this.nbSamplesProcessed.incrementAndGet();
		double[] outArray = new double[3];
		double sum = bias;
		for (int i = 0; i < convXSize; i++) {
			if (startX + i >= 0 && startX + i < dataXSize) {
				for (int j = 0; j < convYSize; j++) {
					if (startY + j >= 0 && startY + j < dataYSize) {
						for (int channel = 0; channel < nbChannels; channel++) {
							final int convPos = j * convXSize + i + channel * convXSize * convYSize;
							final int pos = (j + startY) * dataXSize + (i + startX) + channel * dataXSize * dataYSize;
							sum += this.realWeights[convPos] * input[pos];
						}
					}
				}
			}
		}
		double out = Math.min(Math.max(sum, -1), 1);
		// output index: -1->0 0->1, 1->2
		if (this.deterministic) {
			this.accumAgreement.addAndGet(1.0);
			if (out > 0.) {
				outArray[0] = 0.;
				outArray[1] = 0.;
				outArray[2] = 1.;
			} else if (out == 0.) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
			} else {
				outArray[0] = 1.;
				outArray[1] = 0.;
				outArray[2] = 0.;
			}
		} else {
			if (out > 0.) {
				outArray[0] = 0.;
				outArray[1] = 1. - out;
				outArray[2] = out;
				this.accumAgreement.addAndGet(Math.max(outArray[1], outArray[2]));
			} else if (out == 0.) {
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
	public final double[] getWeights() {
		return realWeights;
	}
}
