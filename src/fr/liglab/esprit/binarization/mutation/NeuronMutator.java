package fr.liglab.esprit.binarization.mutation;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import fr.liglab.esprit.binarization.FilesProcessing;
import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.neuron.SumHistogram;
import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.neuron.TernaryOutputNeuron;
import fr.liglab.esprit.binarization.neuron.TernaryWeightsNeuron;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public class NeuronMutator {
	final private int[] sums;
	final private TernaryProbDistrib[] originalNeuronOutput;
	private int maxSum;
	private int minSum;
	final private List<byte[]> input;
	final private TernaryWeightsNeuron binarizedNeuron;
	// final private int twPosMinIndex;
	// final private int twNegMaxIndex;

	// force posneg always active
	public NeuronMutator(final TernaryOutputNeuron originalNeuron, final TernaryWeightsNeuron binarizedNeuron,
			final List<byte[]> input) {
		this.binarizedNeuron = binarizedNeuron;
		this.originalNeuronOutput = new TernaryProbDistrib[input.size()];
		this.input = input;
		for (int i = 0; i < input.size(); i++) {
			this.originalNeuronOutput[i] = originalNeuron.getOutputProbs(input.get(i));
		}
		this.sums = new int[input.size()];
		int tmpMaxSum = Integer.MIN_VALUE;
		int tmpMinSum = Integer.MAX_VALUE;
		for (int sampleIndex = 0; sampleIndex < input.size(); sampleIndex++) {
			byte[] sample = input.get(sampleIndex);
			int sum = binarizedNeuron.getSum(sample);
			this.sums[sampleIndex] = sum;
			tmpMaxSum = Math.max(tmpMaxSum, sum);
			tmpMinSum = Math.min(tmpMinSum, sum);
		}
		this.maxSum = tmpMaxSum + 2;
		this.minSum = tmpMinSum - 2;
	}

	public TernaryConfig getBestConfig(final int mutatedNeuronIndex, final int nbMut) {
		SumHistogram[] histo = getSumDist(mutatedNeuronIndex, nbMut);
		// for (SumHistogram h : histo) {
		// System.out.println(h);
		// }
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
		double minOneWellClassified = histo[0].getSum(this.minSum, tl - 1);
		double zeroWellClassified = histo[1].getSum(tl, th);
		double oneWellClassified = histo[2].getSum(th + 1, this.maxSum);
		double score = minOneWellClassified + zeroWellClassified + oneWellClassified;
		return new TernaryConfig(th, tl, mutatedNeuronIndex, nbMut, score / this.originalNeuronOutput.length);
	}

	public final int getSum(int inputIndex) {
		return this.sums[inputIndex];
	}

	public void applyMutation(final int mutatedNeuronIndex, final int nbMut) {
		int delta = 0;
		final double originalWeight = this.binarizedNeuron.getWeights()[mutatedNeuronIndex];

		if (originalWeight == -1.) {
			if (nbMut == 1) {
				delta = 1;
			} else {
				delta = 2;
			}
		} else if (originalWeight == 0.) {
			if (nbMut == 1) {
				delta = 1;
			} else {
				delta = -1;
			}
		} else if (originalWeight == 1.) {
			if (nbMut == 1) {
				delta = -2;
			} else {
				delta = -1;
			}
		} else {
			throw new RuntimeException("shouldn't be here");
		}
		int tmpMaxSum = Integer.MIN_VALUE;
		int tmpMinSum = Integer.MAX_VALUE;
		for (int i = 0; i < this.sums.length; i++) {
			int sum = this.getSum(i);
			int diff = 0;
			diff = delta * this.input.get(i)[mutatedNeuronIndex];
			this.sums[i] += diff;
			tmpMaxSum = Math.max(tmpMaxSum, this.sums[i]);
			tmpMinSum = Math.min(tmpMinSum, this.sums[i]);
		}
		this.maxSum = tmpMaxSum + 2;
		this.minSum = tmpMinSum - 2;
	}

	private SumHistogram[] getSumDist(final int mutatedNeuronIndex, final int nbMut) {
		int delta = 0;
		if (mutatedNeuronIndex != -1) {
			final double originalWeight = this.binarizedNeuron.getWeights()[mutatedNeuronIndex];

			if (originalWeight == -1.) {
				if (nbMut == 1) {
					delta = 1;
				} else {
					delta = 2;
				}
			} else if (originalWeight == 0.) {
				if (nbMut == 1) {
					delta = 1;
				} else {
					delta = -1;
				}
			} else if (originalWeight == 1.) {
				if (nbMut == 1) {
					delta = -2;
				} else {
					delta = -1;
				}
			} else {
				throw new RuntimeException("shouldn't be here");
			}
		}
		SumHistogram[] s = new SumHistogram[3];
		for (int i = 0; i < s.length; i++) {
			s[i] = new SumHistogram(this.minSum, this.maxSum);
		}
		for (int i = 0; i < this.sums.length; i++) {
			int sum = this.getSum(i);
			int diff = 0;
			if (mutatedNeuronIndex != -1) {
				diff = delta * this.input.get(i)[mutatedNeuronIndex];
			}
			sum += diff;
			for (int j = 0; j < 3; j++) {
				s[j].addOccurence(sum, this.originalNeuronOutput[i].getProbs()[j]);
			}
		}
		return s;
	}

	public static void main(String[] args) throws IOException {
		double[] weights = FilesProcessing
				.getWeights("/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/sw1.txt", 401);
		double bias = FilesProcessing.getBias("/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/sb1.txt",
				401);
		TanHNeuron nOrigin = new TanHNeuron(weights, bias, false);
		List<byte[]> input = FilesProcessing.getAllTrainingSet(
				"/Users/vleroy/workspace/esprit/mnist_binary/MNIST_32_32/dataTrain.txt", Integer.MAX_VALUE);
		TernaryWeightsNeuron nBinarized = new TernaryWeightsNeuron(Arrays.copyOf(weights, weights.length), 32, 7, 2,
				-1);
		NeuronMutator cb = new NeuronMutator(nOrigin, nBinarized, input);
		int nbPosWeights = nBinarized.getNbPosWeights();
		int nbNegWeights = nBinarized.getNbNegWeights();
		// NeuronComparator nc = new NeuronComparator(nOrigin, nBinarized,
		// ScoreFunctions.AGREEMENT);
		// for (byte[] sample : input) {
		// nc.update(sample);
		// }
		double originalScore = cb.getBestConfig(-1, -1).getScore();
		// System.out.println("original neuron comparator score " +
		// nc.getScore() / input.size());
		System.out.println(
				"original score " + originalScore + " raw, " + originalScore / nOrigin.getMaxAgreement() + " relative");
		while (true) {
			double bestMutationDiff = 0.;
			int bestIndex = -1;
			int bestNbMut = -1;
			TernaryConfig bestConf = null;
			for (int mutatedNeuronIndex = 0; mutatedNeuronIndex < weights.length; mutatedNeuronIndex++) {
				for (int nbMut = 1; nbMut < 3; nbMut++) {
					TernaryConfig c = cb.getBestConfig(mutatedNeuronIndex, nbMut);
					double mutationDiff = (c.getScore() - originalScore);
					if (mutationDiff > bestMutationDiff) {
						System.out.println("successful mutation " + mutatedNeuronIndex + " " + nbMut + " "
								+ c.getScore() + " (" + mutationDiff + ")");
						bestIndex = mutatedNeuronIndex;
						bestNbMut = nbMut;
						bestMutationDiff = mutationDiff;
						bestConf = c;
					}
				}
			}
			if (bestIndex == -1) {
				break;
			}
			originalScore = bestConf.getScore();
			System.out.println("best mutation " + bestMutationDiff + " at " + bestIndex);
			System.out.println("original weight " + nBinarized.getWeights()[bestIndex]);
			if (nBinarized.getWeights()[bestIndex] == -1.) {
				if (bestNbMut == 1) {
					nBinarized.getWeights()[bestIndex] = 0.;
				} else {
					nBinarized.getWeights()[bestIndex] = 1;
				}
			} else if (nBinarized.getWeights()[bestIndex] == 0.) {
				if (bestNbMut == 1) {
					nBinarized.getWeights()[bestIndex] = 1.;
				} else {
					nBinarized.getWeights()[bestIndex] = -1;
				}
			} else if (nBinarized.getWeights()[bestIndex] == 1.) {
				if (bestNbMut == 1) {
					nBinarized.getWeights()[bestIndex] = -1.;
				} else {
					nBinarized.getWeights()[bestIndex] = 0;
				}
			} else {
				throw new RuntimeException("what?");
			}
			cb.applyMutation(bestIndex, bestNbMut);
			System.out.println("new weight " + nBinarized.getWeights()[bestIndex]);
		}
		System.out.println(
				"final score " + originalScore + " raw, " + originalScore / nOrigin.getMaxAgreement() + " relative");
		// nBinarized.setTh(bestConf.th);
		// nBinarized.setTl(bestConf.tl);
		// nc = new NeuronComparator(nOrigin, nBinarized,
		// ScoreFunctions.AGREEMENT);
		// for (byte[] sample : input) {
		// nc.update(sample);
		// }
		// System.out.println("new neuron comparator score " + nc.getScore() /
		// input.size());
	}
}
