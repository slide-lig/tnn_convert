package fr.liglab.esprit.binarization;

import java.util.Arrays;

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
		String trainingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		// String outputFile = args[3];
		double[] weights = FilesProcessing.getWeights(weightsData, 0);
		double bias = FilesProcessing.getBias(biasData, 0);
		TernaryOutputNeuron nOrigin = new TanHNeuron(weights, bias, true);
		TernaryOutputNeuron nBinarized = new TernaryWeightsNeuron(Arrays.copyOf(weights, weights.length), 0.034737,
				-0.03607, 2, 3);
		NeuronComparator nc = new NeuronComparator(nOrigin, nBinarized, ScoreFunctions.AGREEMENT);
		for (boolean[] sample : FilesProcessing.getTrainingSet(trainingData, 40)) {
			nc.update(sample);
		}
		System.out.println(nc.getConfMat() + "\nscore=" + nc.getScore());
	}

}
