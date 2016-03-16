package fr.liglab.esprit.binarization;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;

public class BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer {

	public static enum ScoreFunctions {
		AGREEMENT, SQUARED_ERROR, AVG_ACCURACY
	}

	private final double[] realWeights;
	private final double bias;
	private final int[] orderedAbsWeightsIndex;
	private final List<TreeMap<Integer, TernaryProbDistrib>> binarizationQuality;
	private int nbQualityCellsFilled = 0;
	private boolean deterministic = false;
	private ScoreFunctions scoreFun = ScoreFunctions.SQUARED_ERROR;

	public BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer(final double[] weights, final double bias) {
		this.bias = bias;
		List<Integer> sortArray = new ArrayList<>(weights.length);
		for (int i = 0; i < weights.length; i++) {
			sortArray.add(i);
		}
		Collections.sort(sortArray, new Comparator<Integer>() {

			@Override
			public int compare(Integer o1, Integer o2) {
				Double d1 = Math.abs(weights[o1]);
				Double d2 = Math.abs(weights[o2]);
				int ret = d2.compareTo(d1);
				if (ret != 0) {
					return ret;
				} else {
					return o1.compareTo(o2);
				}
			}
		});

		this.realWeights = weights;
		orderedAbsWeightsIndex = new int[sortArray.size()];
		for (int i = 0; i < orderedAbsWeightsIndex.length; i++) {
			orderedAbsWeightsIndex[i] = sortArray.get(i);
		}
		this.binarizationQuality = new ArrayList<>(realWeights.length + 1);
		for (int i = 0; i < realWeights.length + 1; i++) {
			this.binarizationQuality.add(new TreeMap<>());
		}
		// for (int i = 0; i < this.binarizationQuality.length; i++) {
		// for (int j = 0; j < this.binarizationQuality[i].length; j++) {
		// this.binarizationQuality[i][j] = new TernaryProbDistrib();
		// }
		// }
	}

	public void getOutputProbs(boolean[] input, double[] outArray) {

		double sum = bias;
		for (int i = 0; i < input.length; i++) {
			// switch (input[i]) {
			// case -1:
			// sum -= this.realWeights[i];
			// break;
			// case 1:
			// sum += this.realWeights[i];
			// break;
			// case 0:
			// break;
			// default:
			// throw new RuntimeException("input is supposed to be -1,0,1, not "
			// + input[i]);
			// }
			if (input[i]) {
				sum += this.realWeights[i];
			}
		}
		double out = Math.tanh(sum);
		// output index: -1->0 0->1 1->2
		if (this.deterministic) {
			if (out > 0) {
				outArray[0] = 0.;
				outArray[1] = 0.;
				outArray[2] = 1.;
			} else if (out == 0) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
			} else {
				outArray[0] = 1;
				outArray[1] = 0.;
				outArray[2] = 0.;
			}
		} else {
			if (out > 0) {
				outArray[0] = 0.;
				outArray[1] = 1. - out;
				outArray[2] = out;
			} else if (out == 0) {
				outArray[0] = 0.;
				outArray[1] = 1.;
				outArray[2] = 0.;
			} else {
				outArray[0] = -out;
				outArray[1] = 1. + out;
				outArray[2] = 0.;
			}
		}
	}

	public double[] getOutputProbs(boolean[] input) {
		double[] output = new double[3];
		this.getOutputProbs(input, output);
		return output;
	}

	public void updateBinaryAccuracy(boolean[] input, double[] outputProbs) {
		int sum = 0;
		this.updateAgreement(0, sum, outputProbs);
		for (int i = 1; i <= this.realWeights.length; i++) {
			final int nextNonZeroWeight = this.orderedAbsWeightsIndex[i - 1];
			if (input[nextNonZeroWeight]) {
				if (this.realWeights[nextNonZeroWeight] > 0) {
					sum++;
				} else {
					sum--;
				}
			}
			this.updateAgreement(i, sum, outputProbs);
		}
	}

	private void updateAgreement(int binarizationIndex, int sum, double[] outputProbs) {
		TreeMap<Integer, TernaryProbDistrib> map = this.binarizationQuality.get(binarizationIndex);
		TernaryProbDistrib dist = map.get(sum);
		if (dist == null) {
			dist = new TernaryProbDistrib();
			map.put(sum, dist);
			nbQualityCellsFilled++;
		}
		dist.merge(outputProbs);
	}

	public TernarySolution findBestBinarizedConfiguration() {
		// weightBinarizationIndex is exclusive, so weightBinarizationIndex = 0
		// means nothing is binary
		double currentBestScore = Double.NEGATIVE_INFINITY;
		TernarySolution currentBestParam = null;
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
					double score = currentDistrib.getScore(scoreFun);
					if (currentBestParam == null || score > currentBestScore) {
						currentBestScore = score;
						currentBestParam = new TernarySolution(th, tl,
								weightBinarizationIndex == this.realWeights.length ? 1000000.
										: Math.abs(
												this.realWeights[this.orderedAbsWeightsIndex[weightBinarizationIndex]]),
								new TernaryConfusionMatrix(currentDistrib), score);
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
		return currentBestParam;
	}

	private static class TernaryConfusionMatrix {
		private double[][] matrix;

		public TernaryConfusionMatrix() {
			this.matrix = new double[3][3];
			for (double[] line : this.matrix) {
				Arrays.fill(line, 0.);
			}
		}

		public TernaryConfusionMatrix(TernaryConfusionMatrix source) {
			this.matrix = new double[3][3];
			for (int i = 0; i < this.matrix.length; i++) {
				System.arraycopy(source.matrix[i], 0, this.matrix[i], 0, 3);
			}
		}

		// chosenOutput: -1->0 0->1 1->2
		public void add(TernaryProbDistrib distrib, int chosenOutput) {
			double[] line = this.matrix[chosenOutput];
			for (int i = 0; i < line.length; i++) {
				line[i] += distrib.groundTruthProbs[i];
			}
		}

		public void remove(TernaryProbDistrib distrib, int chosenOutput) {
			double[] line = this.matrix[chosenOutput];
			for (int i = 0; i < line.length; i++) {
				line[i] -= distrib.groundTruthProbs[i];

				if (line[i] < 0.) {
					if (line[i] > -0.000000001) {
						line[i] = 0;
					}
				}
			}
		}

		public double getScore(ScoreFunctions sf) {
			switch (sf) {
			case AGREEMENT:
				return this.getAgreement();
			case AVG_ACCURACY:
				return this.getAvgAccuracyByClass();
			case SQUARED_ERROR:
				return this.getMinSumSquaredError();
			default:
				throw new RuntimeException("missing case");

			}
		}

		public double getAgreement() {
			double agreement = 0;
			for (int i = 0; i < this.matrix.length; i++) {
				agreement += this.matrix[i][i];
			}
			return agreement;
		}

		public double getMinSumSquaredError() {
			double sumSquared = this.matrix[0][1] + this.matrix[1][0] + this.matrix[1][2] + this.matrix[2][1]
					+ 4 * (this.matrix[0][2] + this.matrix[2][0]);
			return -sumSquared;
		}

		public double getAvgAccuracyByClass() {
			double avg = 0.;
			int nbDiv = 0;
			double[] sum = new double[3];
			for (int i = 0; i < this.matrix.length; i++) {
				for (int j = 0; j < this.matrix[i].length; j++) {
					sum[j] += this.matrix[i][j];
				}
			}
			for (int i = 0; i < this.matrix.length; i++) {
				if (sum[i] > 0) {
					nbDiv++;
					avg += (this.matrix[i][i] / sum[i]);
				}
			}
			return avg / nbDiv;
		}

		@Override
		public String toString() {
			return Arrays.toString(this.matrix[0]) + Arrays.toString(this.matrix[1]) + Arrays.toString(this.matrix[2]);
		}

	}

	private static class TernaryProbDistrib {
		double[] groundTruthProbs;

		public TernaryProbDistrib() {
			this.groundTruthProbs = new double[3];
			Arrays.fill(this.groundTruthProbs, 0.);
		}

		public void merge(double[] outputProbs) {
			for (int i = 0; i < outputProbs.length; i++) {
				this.groundTruthProbs[i] += outputProbs[i];
			}
		}

		@Override
		public String toString() {
			return "TernaryProbDistrib [groundTruthProbs=" + Arrays.toString(groundTruthProbs) + "]";
		}
	}

	public static class TernarySolution {
		public final int th;
		public final int tl;
		public final double tw;
		public final TernaryConfusionMatrix confusionMat;
		public final double score;

		public TernarySolution(int th, int tl, double tw, TernaryConfusionMatrix confusionMat, double score) {
			super();
			this.th = th;
			this.tl = tl;
			this.tw = tw;
			this.confusionMat = confusionMat;
			this.score = score;
		}

		@Override
		public String toString() {
			return "TernarySolution [th=" + th + ", tl=" + tl + ", tw=" + tw + ", confusionMat=" + confusionMat
					+ ", score=" + score + "]";
		}

	}

	public static void main(String[] args) throws Exception {

		// BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer test = new
		// BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer(new double[]
		// {-0.0051861, -0.02769, -0.0028713, -0.01298, -2.8156E-4, 0.022563,
		// 0.0036143}, 0.1);
		final int neuronIndex = 0;
		// load neuron weights
		BufferedReader br = new BufferedReader(
				new FileReader("/Users/vleroy/workspace/esprit/mnist_binary/Network/w1.txt"));
		String line;
		int lineNumber = 0;
		double[] weights = null;
		while ((line = br.readLine()) != null) {
			if (lineNumber == neuronIndex) {
				String[] input = line.split(",");
				weights = new double[input.length];
				for (int i = 0; i < input.length; i++) {
					weights[i] = Double.parseDouble(input[i]);
				}
				break;
			}
			lineNumber++;
		}
		br.close();
		// Random r = new Random(0);
		// for (int i = 0; i < weights.length; i++) {
		// double w = 0.5 + r.nextDouble() * 0.1;
		// if (i < weights.length / 2) {
		// w = -w;
		// }
		// weights[i] = w;
		// }
		// load bias
		double bias = 0.;
		br = new BufferedReader(new FileReader("/Users/vleroy/workspace/esprit/mnist_binary/Network/b1.txt"));
		lineNumber = 0;
		while ((line = br.readLine()) != null) {
			if (lineNumber == neuronIndex) {
				bias = Double.valueOf(line);
				break;
			}
			lineNumber++;
		}
		br.close();
		BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer transformer = new BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer(
				weights, bias);
		// read input
		br = new BufferedReader(
				new FileReader("/Users/vleroy/workspace/esprit/mnist_binary/MNIST_32_32/dataTrain.txt"));

		double[] probBuffer = new double[3];
		double[] sumProbBaseline = new double[3];
		int nbSamp = 0;
		while ((line = br.readLine()) != null) {
			if (!line.isEmpty()) {
				String[] data = line.split(",");
				boolean[] input = new boolean[data.length];
				for (int i = 0; i < data.length; i++) {
					if (data[i].equals("1")) {
						input[i] = true;
					} else {
						input[i] = false;
					}
				}
				transformer.getOutputProbs(input, probBuffer);
				transformer.updateBinaryAccuracy(input, probBuffer);
				for (int i = 0; i < probBuffer.length; i++) {
					sumProbBaseline[i] += probBuffer[i];
				}
			}
			nbSamp++;
			if (nbSamp % 10000 == 0) {
				System.out.println("sample " + nbSamp);
			}
		}
		br.close();
		System.out.println(transformer.findBestBinarizedConfiguration());
		System.out.println("baseline output: " + Arrays.toString(sumProbBaseline));
		System.out.println("parameter combinations encountered: " + transformer.nbQualityCellsFilled);
	}

}
