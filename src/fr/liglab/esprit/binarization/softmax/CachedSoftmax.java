package fr.liglab.esprit.binarization.softmax;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

public class CachedSoftmax {
	final private int[][] posSums;
	final private int[][] negSums;

	public CachedSoftmax(int[][] posSums, int[][] negSums) {
		super();
		this.posSums = posSums;
		this.negSums = negSums;
	}

	// force posneg always active
	public CachedSoftmax(final double[] originalWeights, final List<byte[]> input) {
		List<Integer> posWeightsIndex = new ArrayList<>(originalWeights.length);
		List<Integer> negWeightsIndex = new ArrayList<>(originalWeights.length);
		for (int i = 0; i < originalWeights.length; i++) {
			if (originalWeights[i] > 0) {
				posWeightsIndex.add(i);
			} else if (originalWeights[i] < 0) {
				negWeightsIndex.add(i);
			}
		}
		if (posWeightsIndex.isEmpty() || negWeightsIndex.isEmpty()) {
			throw new RuntimeException("cannot force pos/neg tw if all weights are positive or negative");
		} else {
			Collections.sort(posWeightsIndex, new Comparator<Integer>() {

				@Override
				public int compare(Integer o1, Integer o2) {
					Double d1 = originalWeights[o1];
					Double d2 = originalWeights[o2];
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
					Double d1 = Math.abs(originalWeights[o1]);
					Double d2 = Math.abs(originalWeights[o2]);
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
			// int tmpMaxSumPos = 0;
			// int tmpMinSumPos = 0;
			// int tmpMaxSumNeg = 0;
			// int tmpMinSumNeg = 0;
			for (int sampleIndex = 0; sampleIndex < input.size(); sampleIndex++) {
				byte[] sample = input.get(sampleIndex);
				int sum = 0;
				Iterator<Integer> indexIter = posWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					sum += sample[indexIter.next()];
					// tmpMaxSumPos = Math.max(tmpMaxSumPos, sum);
					// tmpMinSumPos = Math.min(tmpMinSumPos, sum);
					this.posSums[sampleIndex][i] = sum;
				}
				sum = 0;
				indexIter = negWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					sum -= sample[indexIter.next()];
					// tmpMaxSumNeg = Math.max(tmpMaxSumNeg, sum);
					// tmpMinSumNeg = Math.min(tmpMinSumNeg, sum);
					this.negSums[sampleIndex][i] = sum;
				}
			}
		}
	}

	public static double getCurrentPerf(final CachedSoftmax[] cachedNeurons, final SoftMaxConfig[] existingConfigs,
			final int[] groundTruth) {
		double perf = 0.;
		for (int input = 0; input < groundTruth.length; input++) {
			int[] sums = new int[cachedNeurons.length];
			int bestFalseSum = Integer.MIN_VALUE;
			int nbHavingBestFalseSum = 0;
			for (int neuron = 0; neuron < sums.length; neuron++) {
				sums[neuron] = cachedNeurons[neuron].getSum(input, existingConfigs[neuron].nbPosWeights,
						existingConfigs[neuron].nbNegWeights) + existingConfigs[neuron].getBias();
				if (neuron != groundTruth[input]) {
					if (sums[neuron] > bestFalseSum) {
						bestFalseSum = sums[neuron];
						nbHavingBestFalseSum = 1;
					} else if (sums[neuron] == bestFalseSum) {
						nbHavingBestFalseSum++;
					}
				}
			}
			if (bestFalseSum > sums[groundTruth[input]]) {
				// this one is lost, doesn t depend on this neuron
				// nbBadClassifications++;
			} else if (bestFalseSum == sums[groundTruth[input]]) {
				perf += 1. / (nbHavingBestFalseSum + 1);
			} else {
				perf += 1.;
			}
		}
		return perf;
	}

	public static SoftMaxConfig getBestConfig(final CachedSoftmax[] cachedNeurons,
			final SoftMaxConfig[] existingConfigs, final int[] groundTruth, final int configuredNeuronIndex,
			final int nbPosWeights, final int nbNegWeights) {
		// int nbBadClassifications = 0;
		final int MIN_BIAS = -2000;
		final int MB_BIAS_OPTIONS = 4000;
		int maxSeenBias = Integer.MIN_VALUE;
		int minSeenBias = Integer.MAX_VALUE;
		// 0 is below, 1 is here, 2 is above
		double[][] biasScores = new double[MB_BIAS_OPTIONS][3];
		// int nbMatch = 0;
		// int nbOther = 0;
		for (int input = 0; input < groundTruth.length; input++) {
			int[] sums = new int[cachedNeurons.length];
			int bestFalseSum = Integer.MIN_VALUE;
			int nbHavingBestFalseSum = 0;
			for (int neuron = 0; neuron < sums.length; neuron++) {
				if (neuron != configuredNeuronIndex) {
					sums[neuron] = cachedNeurons[neuron].getSum(input, existingConfigs[neuron].nbPosWeights,
							existingConfigs[neuron].nbNegWeights) + existingConfigs[neuron].getBias();
					if (neuron != groundTruth[input]) {
						if (sums[neuron] > bestFalseSum) {
							bestFalseSum = sums[neuron];
							nbHavingBestFalseSum = 1;
						} else if (sums[neuron] == bestFalseSum) {
							nbHavingBestFalseSum++;
						}
					}
				}
			}
			if (groundTruth[input] == configuredNeuronIndex) {
				// nbMatch++;
				int baseSum = cachedNeurons[configuredNeuronIndex].getSum(input, nbPosWeights, nbNegWeights);
				int biasForEqual = bestFalseSum - baseSum;
				maxSeenBias = Math.max(maxSeenBias, biasForEqual);
				minSeenBias = Math.min(minSeenBias, biasForEqual);
				biasScores[biasForEqual - MIN_BIAS][1] += 1. / (nbHavingBestFalseSum + 1);
				biasScores[biasForEqual - MIN_BIAS][2] += 1.;
			} else {
				// nbOther++;
				if (bestFalseSum > sums[groundTruth[input]]) {
					// this one is lost, doesn t depend on this neuron
					// nbBadClassifications++;
				} else {
					// this neuron is important because good neuron can lead
					// for
					// sum (maybe equal with others)
					int baseSum = cachedNeurons[configuredNeuronIndex].getSum(input, nbPosWeights, nbNegWeights);
					int biasForEqual = sums[groundTruth[input]] - baseSum;
					maxSeenBias = Math.max(maxSeenBias, biasForEqual);
					minSeenBias = Math.min(minSeenBias, biasForEqual);
					if (bestFalseSum == sums[groundTruth[input]]) {
						biasScores[biasForEqual - MIN_BIAS][0] += 1. / (nbHavingBestFalseSum + 1);
						biasScores[biasForEqual - MIN_BIAS][1] += 1. / (nbHavingBestFalseSum + 2);
					} else {
						biasScores[biasForEqual - MIN_BIAS][0] += 1.;
						biasScores[biasForEqual - MIN_BIAS][1] += 0.5;
					}
				}
			}
		}
		maxSeenBias++;
		minSeenBias--;
		double sumBelow = 0.;
		for (int i = minSeenBias - MIN_BIAS; i <= maxSeenBias - MIN_BIAS; i++) {
			sumBelow += biasScores[i][0];
		}
		double currentScore = sumBelow;
		double bestScore = -1.0;
		int bestBias = Integer.MAX_VALUE;
		// StringBuilder sb = new StringBuilder();
		for (int i = minSeenBias - MIN_BIAS; i <= maxSeenBias - MIN_BIAS; i++) {
			currentScore -= biasScores[i][0];
			double score = currentScore + biasScores[i][1];
			currentScore += biasScores[i][2];
			// sb.append(i + "\t" + score + "\n");
			if (score > bestScore) {
				bestScore = score;
				bestBias = i;
			}
		}
		// if (bestBias > 3000) {
		// System.out.println(nbMatch + "\t" + nbOther + "\t" +
		// nbBadClassifications);
		// System.out.println(sb.toString());
		// System.out.println("wtf");
		// }
		return new SoftMaxConfig(bestBias + MIN_BIAS, nbPosWeights, nbNegWeights, bestScore);
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
		return posSum + negSum;
	}

	public int getNbPosPossibilities() {
		return this.posSums[0].length;
	}

	public int getNbNegPossibilities() {
		return this.negSums[0].length;
	}
}
