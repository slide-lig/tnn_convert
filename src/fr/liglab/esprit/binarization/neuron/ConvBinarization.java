package fr.liglab.esprit.binarization.neuron;

import java.util.List;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class ConvBinarization extends AConvBinarization {
	final private List<byte[]> referenceInput;
	final private TernaryOutputNeuron originalNeuron;

	// force posneg always active
	public ConvBinarization(final TernaryOutputNeuron originalNeuron, final short convXSize, final short convYSize,
			final int inputXSize, final int inputYSize, final int nbChannels, final int padding,final byte inputMaxVal,
			final List<byte[]> input, List<byte[]> referenceInput) {
		super(originalNeuron, convXSize, convYSize, inputXSize, inputYSize, nbChannels, padding, inputMaxVal, input);
		if (referenceInput == null) {
			referenceInput = input;
		}
		this.referenceInput = referenceInput;
		this.originalNeuron = originalNeuron;
	}

	@Override
	TernaryProbDistrib getOriginalOutput(int inputId, int x, int y) {
		TernaryProbDistrib originalOut = this.originalNeuron.getConvOutputProbs(this.referenceInput.get(inputId), x, y,
				this.inputXSize, this.inputYSize, this.convXSize, this.convYSize, this.nbChannels);
		return originalOut;
	}

}
