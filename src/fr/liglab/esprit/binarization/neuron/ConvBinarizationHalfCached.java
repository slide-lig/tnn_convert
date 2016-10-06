package fr.liglab.esprit.binarization.neuron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public class ConvBinarizationHalfCached implements IBinarization {
	final private int[] wIndex;
	final private List<byte[]> input;
	final private List<TernaryProbDistrib[][]> originalOutput;
	// final private List<byte[]> referenceInput;
	final private short convXSize;
	final private short convYSize;
	final private int inputXSize;
	final private int inputYSize;
	final private short maxSum;
	final private short minSum;
	// final private TernaryOutputNeuron originalNeuron;
	final private int nbPosWeights;
	final private int nbNegWeights;
	final private int nbOccurencesOfConv;
	// final private short maxSum;
	// final private short minSum;
	// final private int inputSize;
	// final private int twPosMinIndex;
	// final private int twNegMaxIndex;

	// force posneg always active
	public ConvBinarizationHalfCached(final TernaryOutputNeuron originalNeuron, final short convXSize,
			final short convYSize, final int inputXSize, final int inputYSize, final byte inputMaxVal,
			final List<byte[]> input, List<byte[]> referenceInput) {
		this.input = input;
		// this.originalNeuron = originalNeuron;
		this.convXSize = convXSize;
		this.convYSize = convYSize;
		this.inputXSize = inputXSize;
		this.inputYSize = inputYSize;
		this.maxSum = (short) (convXSize * convYSize * inputMaxVal);
		this.minSum = (short) -this.maxSum;
		if (referenceInput == null) {
			referenceInput = input;
		}
		this.nbOccurencesOfConv = (this.inputXSize - this.convXSize + 1) * (this.inputYSize - this.convYSize + 1)
				* input.size();
		this.originalOutput = new ArrayList<>(referenceInput.size());
		Iterator<byte[]> refDataIter = referenceInput.iterator();
		while (refDataIter.hasNext()) {
			final byte[] refData = refDataIter.next();
			final TernaryProbDistrib[][] outputMat = new TernaryProbDistrib[(this.inputXSize - this.convXSize
					+ 1)][(this.inputYSize - this.convYSize + 1)];
			for (int x = 0; x < outputMat.length; x++) {
				for (int y = 0; y < outputMat[x].length; y++) {
					outputMat[x][y] = originalNeuron.getConvOutputProbs(refData, x, y, this.inputXSize, this.convXSize,
							this.convYSize);
				}
			}
			this.originalOutput.add(outputMat);
		}
		List<Integer> posWeightsIndex = new ArrayList<>(originalNeuron.getWeights().length);
		List<Integer> negWeightsIndex = new ArrayList<>(originalNeuron.getWeights().length);
		for (int i = 0; i < originalNeuron.getWeights().length; i++) {
			if (originalNeuron.getWeightSign(i) > 0) {
				posWeightsIndex.add(i);
			} else if (originalNeuron.getWeightSign(i) < 0) {
				negWeightsIndex.add(i);
			}
		}
		if (posWeightsIndex.isEmpty() || negWeightsIndex.isEmpty()) {
			throw new RuntimeException("cannot force pos/neg tw if all weights are positive or negative");
		} else {
			Collections.sort(posWeightsIndex, new Comparator<Integer>() {

				@Override
				public int compare(Integer o1, Integer o2) {
					Double d1 = originalNeuron.getWeights()[o1];
					Double d2 = originalNeuron.getWeights()[o2];
					int ret = d2.compareTo(d1);
					if (ret != 0) {
						return ret;
					} else {
						return o1.compareTo(o2);
					}
				}
			});
			Collections.sort(negWeightsIndex, new Comparator<Integer>() {

				@Override
				public int compare(Integer o1, Integer o2) {
					Double d1 = Math.abs(originalNeuron.getWeights()[o1]);
					Double d2 = Math.abs(originalNeuron.getWeights()[o2]);
					int ret = d2.compareTo(d1);
					if (ret != 0) {
						return ret;
					} else {
						return o1.compareTo(o2);
					}
				}
			});
			this.nbPosWeights = posWeightsIndex.size();
			this.nbNegWeights = negWeightsIndex.size();
			this.wIndex = new int[posWeightsIndex.size() + negWeightsIndex.size()];
			for (int i = 0; i < posWeightsIndex.size(); i++) {
				this.wIndex[posWeightsIndex.get(i)] = i + 1;
			}
			for (int i = 0; i < negWeightsIndex.size(); i++) {
				this.wIndex[negWeightsIndex.get(i)] = -(i + 1);
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.liglab.esprit.binarization.neuron.IBinarization#getBestConfig(int,
	 * int)
	 */
	@Override
	public TernaryConfig getBestConfig(int nbPosWeights, int nbNegWeights) {
		// System.out.println("getBestConfig " + nbPosWeights + " " +
		// nbNegWeights);
		SumHistogram[] histo = getSumDist(nbPosWeights, nbNegWeights);
		int bestTh = 0;
		int bestTl = 0;
		double tpMinOne = 0.;
		double tpOne = histo[2].getSum();
		double bestAgreement = -1.;
		for (int tl = 0; tl <= histo[0].getDist().length; tl++) {
			double tpZero = 0.;
			double backTpOne = tpOne;
			for (int th = tl - 1; th < histo[0].getDist().length; th++) {
				if (th != tl - 1) {
					tpZero += histo[1].getDist()[th];
					tpOne -= histo[2].getDist()[th];
				}
				// compute quality overall
				double overallAgreement = tpMinOne + tpOne + tpZero;
				if (overallAgreement > bestAgreement) {
					bestAgreement = overallAgreement;
					bestTh = th;
					bestTl = tl;
				}
			}
			tpOne = backTpOne;
			if (tl != histo[0].getDist().length) {
				tpMinOne += histo[0].getDist()[tl];
				tpOne -= histo[2].getDist()[tl];
			}
		}
		return new TernaryConfig(bestTh - histo[0].getOffset(), bestTl - histo[0].getOffset(), nbPosWeights,
				nbNegWeights, bestAgreement / nbOccurencesOfConv);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.liglab.esprit.binarization.neuron.IBinarization#getNbPosPossibilities(
	 * )
	 */
	@Override
	public int getNbPosPossibilities() {
		return this.nbPosWeights;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.liglab.esprit.binarization.neuron.IBinarization#getNbNegPossibilities(
	 * )
	 */
	@Override
	public int getNbNegPossibilities() {
		return this.nbNegWeights;
	}

	private SumHistogram[] getSumDist(int nbPosWeights, int nbNegWeights) {
		final SumHistogram[] s = new SumHistogram[3];
		for (int i = 0; i < s.length; i++) {
			s[i] = new SumHistogram(this.minSum, this.maxSum);
		}
		Iterator<byte[]> dataIter = this.input.iterator();
		Iterator<TernaryProbDistrib[][]> refDataIter = this.originalOutput.iterator();
		while (dataIter.hasNext()) {
			byte[] data = dataIter.next();
			TernaryProbDistrib[][] refData = refDataIter.next();
			for (int x = 0; x < (this.inputXSize - this.convXSize + 1); x++) {
				for (int y = 0; y < (this.inputYSize - this.convYSize + 1); y++) {
					TernaryProbDistrib originalOut = refData[x][y];
					int outputVal = 0;
					for (int i = 0; i < this.convXSize; i++) {
						for (int j = 0; j < this.convYSize; j++) {
							final int convPos = i * this.convXSize + j;
							final int pos = (i + x) * this.inputXSize + (j + y);
							if (this.wIndex[convPos] > 0 && this.wIndex[convPos] <= nbPosWeights) {
								outputVal += data[pos];
							} else if (this.wIndex[convPos] < 0 && this.wIndex[convPos] >= -nbNegWeights) {
								outputVal -= data[pos];
							}
						}
					}
					for (int o = 0; o < 3; o++) {
						s[o].addOccurence(outputVal, originalOut.getProb(0));
					}
				}
			}
		}
		return s;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.liglab.esprit.binarization.neuron.IBinarization#getInputSize()
	 */
	@Override
	public final int getInputSize() {
		return this.input.size();
	}

}
