package fr.liglab.esprit.binarization.neuron;

import java.util.Arrays;
import java.util.BitSet;

import fr.liglab.esprit.binarization.TernaryProbDistrib;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class GroundTruthNeuron implements TernaryOutputNeuron {
	private final BitSet matchPositions;
	private int index;
	private final double[] weights;

	public GroundTruthNeuron(double[] realWeights, int[] outputs, int match) {
		this.weights = realWeights;
		this.matchPositions = new BitSet(outputs.length);
		for (int i = 0; i < outputs.length; i++) {
			if (outputs[i] == match) {
				this.matchPositions.set(i);
			}
		}
		index = 0;
	}

	@Override
	public TernaryProbDistrib getOutputProbs(byte[] input) {
		double[] outArray = new double[3];
		Arrays.fill(outArray, 0.);
		if (this.matchPositions.get(index)) {
			outArray[2] = 1.;
		} else {
			outArray[0] = 1.;
		}
		index++;
		return new TernaryProbDistrib(outArray);
	}

	@Override
	public TernaryProbDistrib getConvOutputProbs(byte[] input, int startX, int startY, int dataXSize, short convXSize,
			short convYSize) {
		throw new NotImplementedException();
	}

	@Override
	public double[] getWeights() {
		return this.weights;
	}

}
