package fr.liglab.esprit.binarization.neuron;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import fr.liglab.esprit.binarization.FilesProcessing;
import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public class CachedBinarization implements IBinarization {
	final private short[][] posSums;
	final private short[][] negSums;
	final private TernaryProbDistrib[] originalNeuronOutput;
	final private short maxSum;
	final private short minSum;
	final private int inputSize;
	// final private int twPosMinIndex;
	// final private int twNegMaxIndex;

	// force posneg always active
	public CachedBinarization(final TernaryOutputNeuron originalNeuron, final List<byte[]> input,
			List<byte[]> referenceInput) {
		if (referenceInput == null) {
			referenceInput = input;
		}
		this.inputSize = input.size();
		this.originalNeuronOutput = new TernaryProbDistrib[referenceInput.size()];
		for (int i = 0; i < referenceInput.size(); i++) {
			this.originalNeuronOutput[i] = originalNeuron.getOutputProbs(referenceInput.get(i));
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
			this.posSums = new short[posWeightsIndex.size()][inputSize];
			this.negSums = new short[negWeightsIndex.size()][inputSize];
			short tmpMaxSumPos = 0;
			short tmpMinSumPos = 0;
			short tmpMaxSumNeg = 0;
			short tmpMinSumNeg = 0;
			for (int sampleIndex = 0; sampleIndex < inputSize; sampleIndex++) {
				byte[] sample = input.get(sampleIndex);
				short sum = 0;
				Iterator<Integer> indexIter = posWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					sum += sample[indexIter.next()];
					if (sum > tmpMaxSumPos) {
						tmpMaxSumPos = sum;
					}
					if (sum < tmpMinSumPos) {
						tmpMinSumPos = sum;
					}
					// tmpMaxSumPos = Math.max(tmpMaxSumPos, sum);
					// tmpMinSumPos = Math.min(tmpMinSumPos, sum);
					this.posSums[i][sampleIndex] = sum;
				}
				sum = 0;
				indexIter = negWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					sum -= sample[indexIter.next()];
					if (sum > tmpMaxSumNeg) {
						tmpMaxSumNeg = sum;
					}
					if (sum < tmpMinSumNeg) {
						tmpMinSumNeg = sum;
					}
					// tmpMaxSumNeg = Math.max(tmpMaxSumNeg, sum);
					// tmpMinSumNeg = Math.min(tmpMinSumNeg, sum);
					this.negSums[i][sampleIndex] = sum;
				}
			}
			this.maxSum = (short) (tmpMaxSumPos + tmpMaxSumNeg);
			this.minSum = (short) (tmpMinSumPos + tmpMinSumNeg);
		}
	}

	public final short[][] getPosSums() {
		return this.posSums;
	}

	public final short[][] getNegSums() {
		return this.negSums;
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
		SumHistogram[] histo = getSumDist(nbPosWeights, nbNegWeights);
		// int tlLB = 0;
		// for (; tlLB < histo[0].getDist().length && histo[0].getDist()[tlLB]
		// >= histo[1].getDist()[tlLB]
		// && histo[0].getDist()[tlLB] >= histo[2].getDist()[tlLB]; tlLB++) {
		// }
		// if (tlLB != 0) {
		// tlLB--;
		// }
		// int thUB = histo[0].getDist().length - 1;
		// for (; thUB >= 0 && histo[2].getDist()[thUB] >=
		// histo[1].getDist()[thUB]
		// && histo[2].getDist()[thUB] >= histo[0].getDist()[thUB]; thUB--) {
		// }
		// if (thUB != histo[0].getDist().length - 1) {
		// thUB++;
		// }
		int bestTh = 0;
		int bestTl = 0;
		double tpMinOne = 0.;
		double tpOne = histo[2].getSum();
		double bestAgreement = -1.;
		for (int tl = 0; tl <= histo[0].getDist().length; tl++) {
			// if (tl >= tlLB) {
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
					// System.out.println((bestTh - histo[0].getOffset()) +
					// " " + (bestTl - histo[0].getOffset()) + " "
					// + bestAgreement + " " + tpMinOne + " " + tpZero + " "
					// + tpOne);
				}
			}
			tpOne = backTpOne;
			// }
			if (tl != histo[0].getDist().length) {
				tpMinOne += histo[0].getDist()[tl];
				tpOne -= histo[2].getDist()[tl];
			}
		}
		return new TernaryConfig(bestTh - histo[0].getOffset(), bestTl - histo[0].getOffset(), nbPosWeights,
				nbNegWeights, bestAgreement / this.originalNeuronOutput.length);
	}

	public final int getSum(int inputIndex, int nbPosWeights, int nbNegWeights) {
		int posSum = 0;
		if (nbPosWeights > 0) {
			posSum = this.posSums[nbPosWeights - 1][inputIndex];
		}
		int negSum = 0;
		if (nbNegWeights > 0) {
			negSum = this.negSums[nbNegWeights - 1][inputIndex];
		}
		return posSum + negSum;
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
		return this.posSums.length;
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
		return this.negSums.length;
	}

	private SumHistogram[] getSumDist(int nbPosWeights, int nbNegWeights) {
		SumHistogram[] s = new SumHistogram[3];
		for (int i = 0; i < s.length; i++) {
			s[i] = new SumHistogram(this.minSum, this.maxSum);
		}
		short[] sums;
		if (nbPosWeights == 0) {
			if (nbNegWeights == 0) {
				sums = new short[inputSize];
			} else {
				sums = this.negSums[nbNegWeights - 1];
			}
		} else {
			if (nbNegWeights == 0) {
				sums = this.posSums[nbPosWeights - 1];
			} else {
				short[] pos = this.posSums[nbPosWeights - 1];
				short[] neg = this.negSums[nbNegWeights - 1];
				sums = new short[inputSize];
				for (int i = 0; i < inputSize; i++) {
					sums[i] = (short) (pos[i] + neg[i]);
				}
			}
		}
		for (int i = 0; i < sums.length; i++) {
			for (int j = 0; j < 3; j++) {
				s[j].addOccurence(sums[i], this.originalNeuronOutput[i].getProb(j));
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
		return inputSize;
	}

	public static void main(String[] args) throws IOException {
		double[] weights = FilesProcessing.getWeights("/Users/vleroy/Desktop/neuron26.txt", 0);
		double bias = FilesProcessing.getBias("/Users/vleroy/Desktop/bias26.txt", 0);
		TernaryOutputNeuron nOrigin = new TanHNeuron(weights, bias, false);
		List<byte[]> input = FilesProcessing.getAllTrainingSet(
				"/Users/vleroy/workspace/esprit/mnist_binary/MNIST_32_32/dataTrain.txt", Integer.MAX_VALUE);
		IBinarization cb = new CachedBinarization(nOrigin, input, null);
		long startTime = System.currentTimeMillis();
		TernaryConfig conf = null;
		for (int i = 0; i < 10000; i++) {
			conf = cb.getBestConfig(38, 37);
		}
		System.out.println("time : " + (System.currentTimeMillis() - startTime));
		System.out.println(conf);
		// TernaryWeightsNeuron nBinarized = new
		// TernaryWeightsNeuron(Arrays.copyOf(weights, weights.length), 0.10321,
		// -0.11495, 1, -2);
		// int nbPosWeights = nBinarized.getNbPosWeights();
		// int nbNegWeights = nBinarized.getNbNegWeights();
		// System.out.println("pos " + nbPosWeights + " neg " + nbNegWeights);
		// for (int i = 0; i < input.size(); i++) {
		// int sumCached = cb.getSum(i, nbPosWeights, nbNegWeights);
		// int refSum = nBinarized.getSum(input.get(i));
		// if (sumCached != refSum) {
		// System.err.println(i + " difference " + sumCached + " vs " + refSum);
		// }
		// }
		// System.out.println(cb.getBestConfig(90, 54));
		// System.out.println(cb.getBestConfig(35, 8));
	}
}
