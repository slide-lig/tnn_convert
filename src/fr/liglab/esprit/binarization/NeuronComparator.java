package fr.liglab.esprit.binarization;

import java.util.Arrays;
import java.util.Random;

import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.neuron.TernaryOutputNeuron;
import fr.liglab.esprit.binarization.neuron.TernaryWeightsNeuron;
import fr.liglab.esprit.binarization.transformer.TernaryConfusionMatrix;

public class NeuronComparator {
	private final TernaryOutputNeuron n1;
	private final TernaryOutputNeuron n2;
	private final ScoreFunctions scoreFun;
	private final TernaryConfusionMatrix confMat;

	public NeuronComparator(TernaryOutputNeuron n1, TernaryOutputNeuron n2, ScoreFunctions scoreFun) {
		super();
		this.n1 = n1;
		this.n2 = n2;
		this.scoreFun = scoreFun;
		this.confMat = new TernaryConfusionMatrix();
	}

	public void update(boolean[] input) {
		TernaryProbDistrib output1 = n1.getOutputProbs(input);
		TernaryProbDistrib output2 = n2.getOutputProbs(input);
		for (int i = 0; i < 3; i++) {
			this.confMat.add(output1, i, output2.getProbs()[i]);
		}
	}

	protected final TernaryOutputNeuron getN1() {
		return n1;
	}

	protected final TernaryOutputNeuron getN2() {
		return n2;
	}

	protected final ScoreFunctions getScoreFun() {
		return scoreFun;
	}

	protected final TernaryConfusionMatrix getConfMat() {
		return confMat;
	}

	protected final double getScore() {
		return this.confMat.getScore(this.scoreFun);
	}

	public static void main(String[] args) throws Exception {
		String testingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		// String outputFile = args[3];
		double[] weights = FilesProcessing.getWeights(weightsData, 1);
		double bias = FilesProcessing.getBias(biasData, 1);
		TernaryOutputNeuron nOrigin = new TanHNeuron(weights, bias, false);
		TernaryOutputNeuron nBinarized = new TernaryWeightsNeuron(Arrays.copyOf(weights, weights.length), 0.012217,
				-0.013479, 19, -16);
		System.out.println(
				nBinarized.getNbPosWeights() + " pos weights, " + nBinarized.getNbNegWeights() + " neg weights");
		NeuronComparator nc = new NeuronComparator(nOrigin, nBinarized, ScoreFunctions.AGREEMENT);
		double[][] pixelFreq = null;
		int nbSamples = 0;
		for (boolean[] sample : FilesProcessing.getAllTrainingSet(testingData, Integer.MAX_VALUE)) {
			if (pixelFreq == null) {
				pixelFreq = new double[sample.length][3];
			}
			nc.update(sample);
			for (int i = 0; i < sample.length; i++) {
				if (sample[i]) {
					double[] dist = nOrigin.getOutputProbs(sample).getProbs();
					for (int output = 0; output < dist.length; output++) {
						pixelFreq[i][output] += dist[output];
					}
				}
			}
			nbSamples++;
		}
		System.out.println(nc.getConfMat() + "\nscore=" + nc.getScore());
		int[][] sumDist = new int[2000][3];
		for (int run = 0; run < 1000000; run++) {
			for (int output = 0; output < 3; output++) {
				int sum = 0;
				for (int i = 0; i < nBinarized.getWeights().length; i++) {
					if (Math.random() < (pixelFreq[i][output] / nbSamples)) {
						sum += nBinarized.getWeights()[i];
					}
				}
				sumDist[sum + 1000][output]++;
			}
		}
		for (int i = 0; i < sumDist.length; i++) {
			boolean write = false;
			for (int j = 0; !write && j < sumDist[i].length; j++) {
				if (sumDist[i][j] != 0) {
					write = true;
				}
			}
			if (write) {
				System.out.println((i - 1000) + "\t" + sumDist[i][0] + "\t" + sumDist[i][1] + "\t" + sumDist[i][2]);
			}
		}
	}

}
