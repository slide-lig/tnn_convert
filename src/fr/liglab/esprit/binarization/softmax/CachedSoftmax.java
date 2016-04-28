package fr.liglab.esprit.binarization.softmax;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

public class CachedSoftmax {
	final private int[][] posSums;
	final private int[][] negSums;
	final private int inputSize;
	private SoftMaxConfig cachedConfig;
	private int[] cachedSums;

	public CachedSoftmax(final int[][] posSums, final int[][] negSums, final int inputSize) {
		super();
		this.posSums = posSums;
		this.negSums = negSums;
		this.inputSize = inputSize;
	}

	// force posneg always active
	public CachedSoftmax(final double[] originalWeights, final List<byte[]> input) {
		this.inputSize = input.size();
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
			this.posSums = new int[posWeightsIndex.size()][inputSize];
			this.negSums = new int[negWeightsIndex.size()][inputSize];
			// int tmpMaxSumPos = 0;
			// int tmpMinSumPos = 0;
			// int tmpMaxSumNeg = 0;
			// int tmpMinSumNeg = 0;
			for (int sampleIndex = 0; sampleIndex < inputSize; sampleIndex++) {
				byte[] sample = input.get(sampleIndex);
				int sum = 0;
				Iterator<Integer> indexIter = posWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					sum += sample[indexIter.next()];
					// tmpMaxSumPos = Math.max(tmpMaxSumPos, sum);
					// tmpMinSumPos = Math.min(tmpMinSumPos, sum);
					this.posSums[i][sampleIndex] = sum;
				}
				sum = 0;
				indexIter = negWeightsIndex.iterator();
				for (int i = 0; indexIter.hasNext(); i++) {
					sum -= sample[indexIter.next()];
					// tmpMaxSumNeg = Math.max(tmpMaxSumNeg, sum);
					// tmpMinSumNeg = Math.min(tmpMinSumNeg, sum);
					this.negSums[i][sampleIndex] = sum;
				}
			}
		}
	}

	public static double getCurrentPerf(final CachedSoftmax[] cachedNeurons, final SoftMaxConfig[] existingConfigs,
			final int[] groundTruth) {
		double perf = 0.;
		int[][] inputSums = new int[cachedNeurons.length][];
		for (int neuron = 0; neuron < inputSums.length; neuron++) {
			inputSums[neuron] = cachedNeurons[neuron].getSums(existingConfigs[neuron]);
		}
		for (int input = 0; input < groundTruth.length; input++) {
			int bestFalseSum = Integer.MIN_VALUE;
			int nbHavingBestFalseSum = 0;
			for (int neuron = 0; neuron < cachedNeurons.length; neuron++) {
				final int sum = inputSums[neuron][input];
				if (neuron != groundTruth[input]) {
					if (sum > bestFalseSum) {
						bestFalseSum = sum;
						nbHavingBestFalseSum = 1;
					} else if (sum == bestFalseSum) {
						nbHavingBestFalseSum++;
					}
				}
			}
			int groundTruthSum = inputSums[groundTruth[input]][input];
			if (bestFalseSum > groundTruthSum) {
				// this one is lost, doesn t depend on this neuron
				// nbBadClassifications++;
			} else if (bestFalseSum == groundTruthSum) {
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
		int[][] inputSums = new int[cachedNeurons.length][];
		for (int neuron = 0; neuron < inputSums.length; neuron++) {
			if (neuron != configuredNeuronIndex) {
				inputSums[neuron] = cachedNeurons[neuron].getSums(existingConfigs[neuron]);
			} else {
				inputSums[neuron] = cachedNeurons[neuron].getSums(nbPosWeights, nbNegWeights);
			}
		}
		for (int input = 0; input < groundTruth.length; input++) {
			int bestFalseSum = Integer.MIN_VALUE;
			int nbHavingBestFalseSum = 0;
			for (int neuron = 0; neuron < cachedNeurons.length; neuron++) {
				if (neuron != configuredNeuronIndex) {
					final int sum = inputSums[neuron][input];
					if (neuron != groundTruth[input]) {
						if (sum > bestFalseSum) {
							bestFalseSum = sum;
							nbHavingBestFalseSum = 1;
						} else if (sum == bestFalseSum) {
							nbHavingBestFalseSum++;
						}
					}
				}
			}
			if (groundTruth[input] == configuredNeuronIndex) {
				// nbMatch++;
				int baseSum = inputSums[configuredNeuronIndex][input];
				int biasForEqual = bestFalseSum - baseSum;
				maxSeenBias = Math.max(maxSeenBias, biasForEqual);
				minSeenBias = Math.min(minSeenBias, biasForEqual);
				biasScores[biasForEqual - MIN_BIAS][1] += 1. / (nbHavingBestFalseSum + 1);
				biasScores[biasForEqual - MIN_BIAS][2] += 1.;
			} else {
				// nbOther++;
				int groundTruthSum = inputSums[groundTruth[input]][input];
				if (bestFalseSum > groundTruthSum) {
					// this one is lost, doesn t depend on this neuron
					// nbBadClassifications++;
				} else {
					// this neuron is important because good neuron can lead
					// for
					// sum (maybe equal with others)
					int baseSum = inputSums[configuredNeuronIndex][input];
					int biasForEqual = groundTruthSum - baseSum;
					maxSeenBias = Math.max(maxSeenBias, biasForEqual);
					minSeenBias = Math.min(minSeenBias, biasForEqual);
					if (bestFalseSum == groundTruthSum) {
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

	// public final int getSum(int inputIndex, int nbPosWeights, int
	// nbNegWeights) {
	// int posSum = 0;
	// if (nbPosWeights > 0) {
	// posSum = this.posSums[inputIndex][nbPosWeights - 1];
	// }
	// int negSum = 0;
	// if (nbNegWeights > 0) {
	// negSum = this.negSums[inputIndex][nbNegWeights - 1];
	// }
	// return posSum + negSum;
	// }

	public final int[] getSums(int nbPosWeights, int nbNegWeights) {
		int[] sums;
		if (nbPosWeights == 0) {
			if (nbNegWeights == 0) {
				sums = new int[inputSize];
			} else {
				sums = Arrays.copyOf(this.negSums[nbNegWeights - 1], this.negSums[nbNegWeights - 1].length);
			}
		} else {
			if (nbNegWeights == 0) {
				sums = Arrays.copyOf(this.posSums[nbPosWeights - 1], this.posSums[nbPosWeights - 1].length);
			} else {
				int[] pos = this.posSums[nbPosWeights - 1];
				int[] neg = this.negSums[nbNegWeights - 1];
				sums = new int[inputSize];
				for (int i = 0; i < inputSize; i++) {
					sums[i] = pos[i] + neg[i];
				}
			}
		}
		return sums;
	}

	public final int[] getSums(SoftMaxConfig conf) {
		if (this.cachedConfig == null || !this.cachedConfig.equals(conf)) {
			this.cachedConfig = conf;
			this.cachedSums = this.getSums(conf.nbPosWeights, conf.nbNegWeights);
			for (int i = 0; i < this.inputSize; i++) {
				this.cachedSums[i] += conf.getBias();
			}
		}
		return this.cachedSums;
	}

	public int getNbPosPossibilities() {
		return this.posSums.length;
	}

	public int getNbNegPossibilities() {
		return this.negSums.length;
	}
}
