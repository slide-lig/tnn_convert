package fr.liglab.esprit.binarization.transformer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;

import fr.liglab.esprit.binarization.FilesProcessing;
import fr.liglab.esprit.binarization.ScoreFunctions;
import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.neuron.TernaryOutputNeuron;

public class SymBinarizer implements TernaryNeuronBinarizer {

	private final TernaryOutputNeuron realNeuron;
	private final int[] orderedAbsWeightsIndex;
	private final List<TreeMap<Integer, TernaryProbDistrib>> binarizationQuality;
	private int nbQualityCellsFilled = 0;

	public SymBinarizer(final TernaryOutputNeuron neuron) {
		this.realNeuron = neuron;
		List<Integer> sortArray = new ArrayList<>(neuron.getWeights().length);
		for (int i = 0; i < neuron.getWeights().length; i++) {
			sortArray.add(i);
		}
		Collections.sort(sortArray, new Comparator<Integer>() {

			@Override
			public int compare(Integer o1, Integer o2) {
				Double d1 = Math.abs(neuron.getWeights()[o1]);
				Double d2 = Math.abs(neuron.getWeights()[o2]);
				int ret = d2.compareTo(d1);
				if (ret != 0) {
					return ret;
				} else {
					return o1.compareTo(o2);
				}
			}
		});

		orderedAbsWeightsIndex = new int[sortArray.size()];
		for (int i = 0; i < orderedAbsWeightsIndex.length; i++) {
			orderedAbsWeightsIndex[i] = sortArray.get(i);
		}
		this.binarizationQuality = new ArrayList<>(neuron.getWeights().length + 1);
		for (int i = 0; i < neuron.getWeights().length + 1; i++) {
			this.binarizationQuality.add(new TreeMap<>());
		}
	}

	public void update(boolean[] input) {
		TernaryProbDistrib outputProbs = this.realNeuron.getOutputProbs(input);
		int sum = 0;
		this.updateAgreement(0, sum, outputProbs);
		for (int i = 1; i <= this.orderedAbsWeightsIndex.length; i++) {
			final int nextNonZeroWeight = this.orderedAbsWeightsIndex[i - 1];
			if (input[nextNonZeroWeight]) {
				if (this.realNeuron.getWeightSign(nextNonZeroWeight) > 0) {
					sum++;
				} else if (this.realNeuron.getWeightSign(nextNonZeroWeight) < 0) {
					sum--;
				}
			}
			this.updateAgreement(i, sum, outputProbs);
		}
	}

	private void updateAgreement(int binarizationIndex, int sum, TernaryProbDistrib outputProbs) {
		TreeMap<Integer, TernaryProbDistrib> map = this.binarizationQuality.get(binarizationIndex);
		TernaryProbDistrib dist = map.get(sum);
		if (dist == null) {
			dist = new TernaryProbDistrib();
			map.put(sum, dist);
			nbQualityCellsFilled++;
		}
		dist.merge(outputProbs);
	}

	public TernarySolution[] findBestBinarizedConfiguration(ScoreFunctions[] scoreFuns) {
		// weightBinarizationIndex is exclusive, so weightBinarizationIndex = 0
		// means nothing is binary
		double[] currentBestScores = new double[scoreFuns.length];
		Arrays.fill(currentBestScores, Double.NEGATIVE_INFINITY);
		TernarySolution[] currentBestParams = new TernarySolution[scoreFuns.length];
		for (int weightBinarizationIndex = 0; weightBinarizationIndex < this.binarizationQuality
				.size(); weightBinarizationIndex++) {
			TreeMap<Integer, TernaryProbDistrib> map = this.binarizationQuality.get(weightBinarizationIndex);
			// System.out.println(weightBinarizationIndex + "->" +
			// map.keySet());
			// sum>th means we output 1
			// sum<tl means we output -1
			// init lineEndDistrib
			TernaryConfusionMatrix lineEndDistrib = new TernaryConfusionMatrix();
			// with th max and tl min we always output 0
			for (TernaryProbDistrib distrib : map.values()) {
				if (distrib != null) {
					lineEndDistrib.add(distrib, 1);
				}
			}
			Iterator<Entry<Integer, TernaryProbDistrib>> thIterator = map.descendingMap().entrySet().iterator();
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
				Iterator<Entry<Integer, TernaryProbDistrib>> tlIterator = map.headMap(th + 1, true).entrySet()
						.iterator();
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
							currentBestParams[i] = new TernarySolution(th, tl,
									weightBinarizationIndex == this.orderedAbsWeightsIndex.length ? 1000000.
											: Math.abs(this.realNeuron
													.getWeights()[this.orderedAbsWeightsIndex[weightBinarizationIndex]]),
									weightBinarizationIndex == this.orderedAbsWeightsIndex.length ? -1000000.
											: -Math.abs(this.realNeuron
													.getWeights()[this.orderedAbsWeightsIndex[weightBinarizationIndex]]),
									weightBinarizationIndex, weightBinarizationIndex,
									new TernaryConfusionMatrix(currentDistrib), score);
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
		}
		return currentBestParams;
	}

	public static void main(String[] args) throws Exception {
		String trainingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		// String outputFile = args[3];
		final int neuronIndex = 0;
		TernaryNeuronBinarizer binarizer = new SymBinarizer(
				new TanHNeuron(FilesProcessing.getWeights(weightsData, neuronIndex),
						FilesProcessing.getBias(biasData, neuronIndex), true));
		for (boolean[] input : FilesProcessing.getAllTrainingSet(trainingData, Integer.MAX_VALUE)) {
			binarizer.update(input);
		}
		System.out.println(Arrays.toString(binarizer.findBestBinarizedConfiguration(ScoreFunctions.values())));
	}

}
