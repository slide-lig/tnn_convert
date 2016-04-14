package fr.liglab.esprit.binarization;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;

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

	public void update(byte[] input) {
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
		List<double[]> weights = FilesProcessing.getAllWeights(weightsData, Integer.MAX_VALUE);
		List<Double> bias = FilesProcessing.getAllBias(biasData, Integer.MAX_VALUE);
		List<byte[]> samples = FilesProcessing.getAllTrainingSet(testingData, Integer.MAX_VALUE);
		// BufferedReader br1 = new BufferedReader(new FileReader(
		// "/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/binary_agreement_asym.txt"));
		BufferedReader br2 = new BufferedReader(new FileReader(
				"/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/binary_agreement_loglog.txt"));
		for (int i = 0; i < weights.size(); i++) {
			TanHNeuron nOrigin = new TanHNeuron(weights.get(i), bias.get(i), false);
			// String s1 = br1.readLine();
			// String[] sp = s1.split(",");
			// TernaryOutputNeuron nBinarized1 = new TernaryWeightsNeuron(
			// Arrays.copyOf(weights.get(i), weights.get(i).length),
			// Double.parseDouble(sp[0]),
			// Double.parseDouble(sp[1]), Integer.parseInt(sp[2]),
			// Integer.parseInt(sp[3]));
			String s2 = br2.readLine();
			String[] sp2 = s2.split(",");
			TernaryOutputNeuron nBinarized2 = new TernaryWeightsNeuron(
					Arrays.copyOf(weights.get(i), weights.get(i).length), Integer.parseInt(sp2[0]),
					Integer.parseInt(sp2[1]), Integer.parseInt(sp2[2]), Integer.parseInt(sp2[3]));
			// NeuronComparator nc1 = new NeuronComparator(nOrigin, nBinarized1,
			// ScoreFunctions.AGREEMENT);
			NeuronComparator nc2 = new NeuronComparator(nOrigin, nBinarized2, ScoreFunctions.AGREEMENT);
			for (byte[] sample : samples) {
				// nc1.update(sample);
				nc2.update(sample);
			}
			double maxAgreement = nOrigin.getMaxAgreement();
			// double diff = nc1.getScore() / samples.size() - nc2.getScore() /
			// samples.size();
			double ratio = (nc2.getScore() / samples.size()) / maxAgreement;
			if (ratio < 0.97) {
				System.out.println("neuron " + i + ": " + ratio);
			}
		}
		// br1.close();
		br2.close();
	}

}
