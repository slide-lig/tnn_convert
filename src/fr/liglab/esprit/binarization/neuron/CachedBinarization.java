package fr.liglab.esprit.binarization.neuron;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import fr.liglab.esprit.binarization.FilesProcessing;
import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public class CachedBinarization {
	final private int[][] posSums;
	final private int[][] negSums;
	final private TernaryProbDistrib[] originalNeuronOutput;
	final private int maxSum;
	final private int minSum;
	// final private int twPosMinIndex;
	// final private int twNegMaxIndex;

	// force posneg always active
	public CachedBinarization(final TernaryOutputNeuron originalNeuron, final List<boolean[]> input) {
		this.originalNeuronOutput = new TernaryProbDistrib[input.size()];
		for (int i = 0; i < input.size(); i++) {
			this.originalNeuronOutput[i] = originalNeuron.getOutputProbs(input.get(i));
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
			this.posSums = new int[input.size()][posWeightsIndex.size()];
			this.negSums = new int[input.size()][negWeightsIndex.size()];
			int tmpMaxSum = 0;
			int tmpMinSum = 0;
			for (int sampleIndex = 0; sampleIndex < input.size(); sampleIndex++) {
				boolean[] sample = input.get(sampleIndex);
				int sum = 0;
				Iterator<Integer> indexIter = posWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					if (sample[indexIter.next()]) {
						sum++;
						tmpMaxSum = Math.max(tmpMaxSum, sum);
					}
					this.posSums[sampleIndex][i] = sum;
				}
				sum = 0;
				indexIter = negWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					if (sample[indexIter.next()]) {
						sum++;
						tmpMinSum = Math.max(tmpMinSum, sum);
					}
					this.negSums[sampleIndex][i] = sum;
				}
			}
			this.maxSum = tmpMaxSum;
			this.minSum = -tmpMinSum;
		}
	}

	public TernaryConfig getBestConfig(int nbPosWeights, int nbNegWeights) {
		SumHistogram[] histo = getSumDist(nbPosWeights, nbNegWeights);
		// for (SumHistogram h : histo) {
		// System.out.println(h);
		// }
		// TODO start point for search
		int crossMinOneZero = histo[0].findCrossPoint(histo[1]);
		int crossZeroOne = histo[1].findCrossPoint(histo[2]);
		int tl = -666666;// they re not supposed to stay at this value
		int th = -666666;// they re not supposed to stay at this value
		if (crossMinOneZero != Integer.MAX_VALUE && crossZeroOne != Integer.MAX_VALUE) {
			// standard case
			tl = crossMinOneZero;
			th = crossZeroOne - 1;
		} else if (crossMinOneZero == Integer.MAX_VALUE) {
			// i.e. 0 dist always under -1 dist
			// never answer 0
			int crossMinOneOne = histo[0].findCrossPoint(histo[2]);
			if (crossMinOneOne == Integer.MAX_VALUE) {
				// i.e. 1 dist always under -1 dist
				// always answer -1
				tl = this.maxSum + 1;
				th = this.maxSum;
			} else {
				// answer -1 or +1
				tl = crossMinOneOne;
				th = tl - 1;
			}
		} else if (crossZeroOne == Integer.MAX_VALUE) {
			// i.e. 1 dist always under 0 dist
			// never answer 1
			th = this.maxSum;
			tl = crossMinOneZero;
		}
		double score = 0.;
		score += histo[0].getSum(this.minSum, tl - 1);
		score += histo[1].getSum(tl, th);
		score += histo[1].getSum(th + 1, this.maxSum);
		return new TernaryConfig(th, tl, nbPosWeights, nbNegWeights, score / this.originalNeuronOutput.length);
	}

	public final int getSum(int inputIndex, int nbPosWeights, int nbNegWeights) {
		int posSum = 0;
		if (nbPosWeights > 0) {
			posSum = this.posSums[inputIndex][nbPosWeights - 1];
		}
		int negSum = 0;
		if (nbNegWeights > 0) {
			negSum = this.negSums[inputIndex][nbNegWeights - 1];
		}
		return posSum - negSum;
	}

	public int getNbPosPossibilities() {
		return this.posSums[0].length;
	}

	public int getNbNegPossibilities() {
		return this.negSums[0].length;
	}

	private SumHistogram[] getSumDist(int nbPosWeights, int nbNegWeights) {
		SumHistogram[] s = new SumHistogram[3];
		for (int i = 0; i < s.length; i++) {
			s[i] = new SumHistogram(this.minSum, this.maxSum);
		}
		for (int i = 0; i < this.posSums.length; i++) {
			int sum = this.getSum(i, nbPosWeights, nbNegWeights);
			for (int j = 0; j < 3; j++) {
				s[j].addOccurence(sum, this.originalNeuronOutput[i].getProbs()[j]);
			}
			// if (this.originalNeuronOutput[i].getProbs()[0] > 0.) {
			// if (Math.random() < this.originalNeuronOutput[i].getProbs()[0]) {
			// s[0].addOccurence(sum, 1.);
			// }
			// } else if (Math.random() <
			// this.originalNeuronOutput[i].getProbs()[2]) {
			// s[1].addOccurence(sum, 1.);
			// } else {
			// s[2].addOccurence(sum, 1.);
			// }
		}
		return s;
	}

	public static void main(String[] args) throws IOException {
		double[] weights = FilesProcessing
				.getFilteredWeightsSingle("/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/sw1.txt", 1);
		double bias = FilesProcessing.getBias("/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/sb1.txt",
				1);
		TernaryOutputNeuron nOrigin = new TanHNeuron(weights, bias, false);
		List<boolean[]> input = FilesProcessing.getFilteredTrainingSet(
				"/Users/vleroy/workspace/esprit/mnist_binary/MNIST_32_32/dataTrain.txt", Integer.MAX_VALUE);
		CachedBinarization cb = new CachedBinarization(nOrigin, input);
		TernaryWeightsNeuron nBinarized = new TernaryWeightsNeuron(Arrays.copyOf(weights, weights.length), 0.012217,
				-0.013479, 19, -16);
		int nbPosWeights = nBinarized.getNbPosWeights();
		int nbNegWeights = nBinarized.getNbNegWeights();
		// for (int i = 0; i < input.size(); i++) {
		// int sumCached = cb.getSum(i, nbPosWeights, nbNegWeights);
		// int refSum = nBinarized.getSum(input.get(i));
		// // if (sumCached != refSum) {
		// // System.err.println(i + " difference " + sumCached + " vs " +
		// // refSum);
		// // }
		// }
		System.out.println(cb.getBestConfig(nbPosWeights, nbNegWeights));
	}
}
