package fr.liglab.esprit.binarization.neuron;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public abstract class AConvBinarization implements IBinarization {
	final int[] wIndex;
	final List<byte[]> input;
	final short convXSize;
	final short convYSize;
	final int inputXSize;
	final int inputYSize;
	final int nbChannels;
	final short maxSum;
	final short minSum;
	final int nbPosWeights;
	final int nbNegWeights;
	final int nbOccurencesOfConv;
	final int padding;

	// force posneg always active
	public AConvBinarization(final TernaryOutputNeuron originalNeuron, final short convXSize, final short convYSize,
			final int inputXSize, final int inputYSize, final int nbChannels, final int padding, final byte inputMaxVal,
			final List<byte[]> input) {
		this.input = input;
		this.padding = padding;
		// this.originalNeuron = originalNeuron;
		this.convXSize = convXSize;
		this.convYSize = convYSize;
		this.inputXSize = inputXSize;
		this.inputYSize = inputYSize;
		this.nbChannels = nbChannels;
		this.maxSum = (short) (convXSize * convYSize * inputMaxVal * nbChannels);
		this.minSum = (short) -this.maxSum;
		this.nbOccurencesOfConv = (this.inputXSize - this.convXSize + 1 + this.padding * 2)
				* (this.inputYSize - this.convYSize + 1 + this.padding * 2) * input.size();
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
		TernaryConfig out = new TernaryConfig(bestTh - histo[0].getOffset(), bestTl - histo[0].getOffset(),
				nbPosWeights, nbNegWeights, bestAgreement / nbOccurencesOfConv);
		// System.out.println(out);
		return out;
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

	abstract TernaryProbDistrib getOriginalOutput(int inputId, int x, int y);

	private SumHistogram[] getSumDist(int nbPosWeights, int nbNegWeights) {
		final SumHistogram[] s = new SumHistogram[3];
		for (int i = 0; i < s.length; i++) {
			s[i] = new SumHistogram(this.minSum, this.maxSum);
		}
		final Iterator<byte[]> dataIter = this.input.iterator();
		int inputId = 0;
		while (dataIter.hasNext()) {
			final byte[] data = dataIter.next();
			for (int y = -this.padding; y < (this.inputYSize - this.convYSize + 1 + this.padding); y++) {
				for (int x = -this.padding; x < (this.inputXSize - this.convXSize + 1 + this.padding); x++) {
					TernaryProbDistrib originalOut = this.getOriginalOutput(inputId, x, y);
					int outputVal = 0;
					for (int i = 0; i < this.convXSize; i++) {
						if (x + i >= 0 && x + i < this.inputXSize) {
							for (int j = 0; j < this.convYSize; j++) {
								if (y + j >= 0 && y + j < this.inputYSize) {
									for (int channel = 0; channel < this.nbChannels; channel++) {
										final int convPos = j * this.convXSize + i
												+ channel * this.convXSize * this.convYSize;
										final int pos = (j + y) * this.inputXSize + (i + x)
												+ channel * this.inputXSize * this.inputYSize;
										if (this.wIndex[convPos] > 0 && this.wIndex[convPos] <= nbPosWeights) {
											outputVal += data[pos];
										} else if (this.wIndex[convPos] < 0 && this.wIndex[convPos] >= -nbNegWeights) {
											outputVal -= data[pos];
										}
									}
								}
							}
						}
					}
					for (int o = 0; o < 3; o++) {
						s[o].addOccurence(outputVal, originalOut.getProb(o));
					}
				}
			}
			inputId++;
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
