package fr.liglab.esprit.binarization.neuron;

import java.util.Iterator;
import java.util.List;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class ValidatePrecom extends AConvBinarization {
	final List<byte[]> referenceInput;

	// force posneg always active
	public ValidatePrecom(final TernaryOutputNeuron n1, final TernaryOutputNeuron n2, final short convXSize,
			final short convYSize, final int inputXSize, final int inputYSize, final int nbChannels, final int padding,
			final byte inputMaxVal, final List<byte[]> input, List<byte[]> referenceInput) {
		super(n1, convXSize, convYSize, inputXSize, inputYSize, nbChannels, padding, inputMaxVal, input);
		if (referenceInput == null) {
			referenceInput = input;
		}
		this.referenceInput = referenceInput;
		Iterator<byte[]> refDataIter = referenceInput.iterator();
		while (refDataIter.hasNext()) {
			final byte[] refData = refDataIter.next();
			final TernaryProbDistrib[][] outputMat = new TernaryProbDistrib[(this.inputXSize - this.convXSize + 1
					+ 2 * this.padding)][(this.inputYSize - this.convYSize + 1 + 2 * this.padding)];
			for (int y = 0; y < outputMat[0].length; y++) {
				for (int x = 0; x < outputMat.length; x++) {
					TernaryProbDistrib t1 = n1.getConvOutputProbs(refData, x - this.padding, y - this.padding,
							this.inputXSize, this.inputYSize, this.convXSize, this.convYSize, this.nbChannels);
					TernaryProbDistrib t2 = n2.getConvOutputProbs(refData, x - this.padding, y - this.padding,
							this.inputXSize, this.inputYSize, this.convXSize, this.convYSize, this.nbChannels);
					System.out.println("recorded " + t1);
					System.out.println("computed " + t2);
					if (!t1.equals(t2)) {
						System.out.println("not equal");
						// System.exit(-1);
					} else {
						System.out.println("equal");
						// System.exit(-1);
					}
				}
			}
		}
	}

	@Override
	TernaryProbDistrib getOriginalOutput(int inputId, int x, int y) {
		return null;
	}

}
