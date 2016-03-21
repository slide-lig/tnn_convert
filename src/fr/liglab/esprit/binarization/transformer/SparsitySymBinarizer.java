package fr.liglab.esprit.binarization.transformer;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.TreeMap;

import fr.liglab.esprit.binarization.FilesProcessing;
import fr.liglab.esprit.binarization.ScoreFunctions;
import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.neuron.TernaryOutputNeuron;

public class SparsitySymBinarizer implements TernaryNeuronBinarizer {

	private final TernaryOutputNeuron realNeuron;
	private final byte[] binarizedWeights;
	private final double tw;
	private final TreeMap<Integer, TernaryProbDistrib> binarizationQuality;

	public SparsitySymBinarizer(final TernaryOutputNeuron neuron, final double tw) {
		this.realNeuron = neuron;
		this.binarizedWeights = new byte[neuron.getWeights().length];
		for (int i = 0; i < neuron.getWeights().length; i++) {
			if (neuron.getWeights()[i] > tw) {
				binarizedWeights[i] = 1;
			} else if (neuron.getWeights()[i] < -tw) {
				binarizedWeights[i] = -1;
			} else {
				binarizedWeights[i] = 0;
			}
		}
		this.tw = tw;
		this.binarizationQuality = new TreeMap<>();
	}

	public void update(boolean[] input) {
		TernaryProbDistrib outputProbs = this.realNeuron.getOutputProbs(input);
		int sum = 0;
		for (int i = 0; i < input.length; i++) {
			if (input[i]) {
				sum += this.binarizedWeights[i];
			}
		}
		this.updateAgreement(sum, outputProbs);
	}

	private void updateAgreement(int sum, TernaryProbDistrib outputProbs) {
		TernaryProbDistrib dist = this.binarizationQuality.get(sum);
		if (dist == null) {
			dist = new TernaryProbDistrib();
			this.binarizationQuality.put(sum, dist);
		}
		dist.merge(outputProbs);
	}

	public TernarySolution[] findBestBinarizedConfiguration(ScoreFunctions[] scoreFuns) {
		// weightBinarizationIndex is exclusive, so weightBinarizationIndex = 0
		// means nothing is binary
		double[] currentBestScores = new double[scoreFuns.length];
		Arrays.fill(currentBestScores, Double.NEGATIVE_INFINITY);
		TernarySolution[] currentBestParams = new TernarySolution[scoreFuns.length];
		// System.out.println(weightBinarizationIndex + "->" +
		// map.keySet());
		// sum>th means we output 1
		// sum<tl means we output -1
		// init lineEndDistrib
		TernaryConfusionMatrix lineEndDistrib = new TernaryConfusionMatrix();
		// with th max and tl min we always output 0
		for (TernaryProbDistrib distrib : this.binarizationQuality.values()) {
			if (distrib != null) {
				lineEndDistrib.add(distrib, 1);
			}
		}
		Iterator<Entry<Integer, TernaryProbDistrib>> thIterator = this.binarizationQuality.descendingMap().entrySet()
				.iterator();
		boolean lastThIter = false;
		int lastThValue = 0;
		boolean firstThIter = true;
		while (lastThIter || thIterator.hasNext()) {
			int th;
			Entry<Integer, TernaryProbDistrib> thEntry = null;
			if (lastThIter) {
				th = lastThValue;
			} else {
				thEntry = thIterator.next();
				th = thEntry.getKey();
			}
			TernaryConfusionMatrix currentDistrib = new TernaryConfusionMatrix(lineEndDistrib);
			// System.err.println(currentDistrib);
			// System.out.println("th=" + th + " possible tl are " +
			// map.headMap(th + 1, true).keySet());
			Iterator<Entry<Integer, TernaryProbDistrib>> tlIterator = this.binarizationQuality.headMap(th + 1, true)
					.entrySet().iterator();
			boolean lastTlIter = false;
			while (lastTlIter || tlIterator.hasNext()) {
				int tl;
				Entry<Integer, TernaryProbDistrib> tlEntry = null;
				if (lastTlIter) {
					tl = th + 1;
				} else {
					tlEntry = tlIterator.next();
					tl = tlEntry.getKey();
				}
				// System.out.println(th + " " + tl);
				for (int i = 0; i < scoreFuns.length; i++) {
					ScoreFunctions scoreFun = scoreFuns[i];
					double score = currentDistrib.getScore(scoreFun);
					if (currentBestParams[i] == null || score > currentBestScores[i]) {
						currentBestScores[i] = score;
						currentBestParams[i] = new TernarySolution(th, tl, tw, -tw, Integer.MAX_VALUE,
								Integer.MAX_VALUE, new TernaryConfusionMatrix(currentDistrib), score);
					}
				}
				// update currentDistrib
				// when th doesn'change but tl goes up some 0 answers
				// become -1
				if (tlEntry != null) {
					currentDistrib.remove(tlEntry.getValue(), 1);
					currentDistrib.add(tlEntry.getValue(), 0);
				}
				if (firstThIter && !lastTlIter && !tlIterator.hasNext()) {
					lastTlIter = true;
				} else {
					lastTlIter = false;
				}
			}
			// when th goes down and tl still min some 0 answers become
			// 1
			if (thEntry != null) {
				lineEndDistrib.remove(thEntry.getValue(), 1);
				lineEndDistrib.add(thEntry.getValue(), 2);
			}
			if (!lastThIter && !thIterator.hasNext()) {
				lastThIter = true;
				lastThValue = th - 1;
			} else {
				lastThIter = false;
			}
			firstThIter = false;
		}
		return currentBestParams;
	}

	public static void main(String[] args) throws Exception {
		String trainingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		// String outputFile = args[3];
		final int neuronIndex = 0;
		TernaryNeuronBinarizer binarizer = new SparsitySymBinarizer(
				new TanHNeuron(FilesProcessing.getWeights(weightsData, neuronIndex),
						FilesProcessing.getBias(biasData, neuronIndex), false),
				FilesProcessing.getCentileAbsWeight(weightsData, 0.80));
		for (boolean[] input : FilesProcessing.getAllTrainingSet(trainingData, Integer.MAX_VALUE)) {
			binarizer.update(input);
		}
		System.out.println(Arrays.toString(binarizer.findBestBinarizedConfiguration(ScoreFunctions.values())));
	}

}
