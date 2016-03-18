package fr.liglab.esprit.binarization.transformer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import fr.liglab.esprit.binarization.FilesProcessing;
import fr.liglab.esprit.binarization.ScoreFunctions;
import fr.liglab.esprit.binarization.TernaryProbDistrib;
import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.neuron.TernaryOutputNeuron;

public class ASymBinarizer implements TernaryNeuronBinarizer {
	private final TernaryOutputNeuron realNeuron;
	private final int[] orderedAbsWeightsIndex;
	private final Map<TwPair, TreeMap<Integer, TernaryProbDistrib>> binarizationQuality;
	private AtomicInteger nbQualityCellsFilled = new AtomicInteger();
	private final ScoreFunctions scoreFun;
	private final int twPosMinIndex;
	private final int twNegMaxIndex;

	public ASymBinarizer(final TernaryOutputNeuron neuron, boolean forcePosNeg, ScoreFunctions scoreFun) {
		this.scoreFun = scoreFun;
		this.realNeuron = neuron;
		List<Integer> sortArray = new ArrayList<>(this.realNeuron.getWeights().length);
		for (int i = 0; i < this.realNeuron.getWeights().length; i++) {
			sortArray.add(i);
		}
		Collections.sort(sortArray, new Comparator<Integer>() {

			@Override
			public int compare(Integer o1, Integer o2) {
				Double d1 = realNeuron.getWeights()[o1];
				Double d2 = realNeuron.getWeights()[o2];
				int ret = d1.compareTo(d2);
				if (ret != 0) {
					return ret;
				} else {
					return o1.compareTo(o2);
				}
			}
		});
		int firstWeightAboveZeroIndex = -1;
		orderedAbsWeightsIndex = new int[sortArray.size()];
		for (int i = 0; i < orderedAbsWeightsIndex.length; i++) {
			orderedAbsWeightsIndex[i] = sortArray.get(i);
			if (forcePosNeg) {
				if (firstWeightAboveZeroIndex == -1 && this.realNeuron.getWeights()[orderedAbsWeightsIndex[i]] >= 0) {
					firstWeightAboveZeroIndex = i;
				}
			}
		}
		if (forcePosNeg) {
			if (firstWeightAboveZeroIndex == 0 || firstWeightAboveZeroIndex == -1) {
				throw new RuntimeException("cannot force pos/neg tw if all weights are positive or negative");
			}

			this.twPosMinIndex = firstWeightAboveZeroIndex - 1;
			this.twNegMaxIndex = firstWeightAboveZeroIndex;
		} else {
			this.twPosMinIndex = -1;
			this.twNegMaxIndex = Integer.MAX_VALUE;
		}
		this.binarizationQuality = new HashMap<>();
		for (int twPos = twPosMinIndex; twPos < this.orderedAbsWeightsIndex.length; twPos++) {
			for (int twNeg = 0; twNeg <= Math.min(twPos + 1, twNegMaxIndex); twNeg++) {
				this.binarizationQuality.put(new TwPair(twPos, twNeg), new TreeMap<>());
			}
		}
	}

	public void update(boolean[] input) {
		TernaryProbDistrib outputProbs = this.realNeuron.getOutputProbs(input);
		int sumStartLign = 0;
		// when twNeg = 0 and all weights are 0 or 1
		for (int i = twPosMinIndex + 1; i < input.length; i++) {
			if (input[this.orderedAbsWeightsIndex[i]]) {
				sumStartLign++;
			}
		}
		for (int twPos = twPosMinIndex; twPos < this.orderedAbsWeightsIndex.length; twPos++) {
			if (twPos >= 0) {
				if (input[this.orderedAbsWeightsIndex[twPos]]) {
					sumStartLign--;
				}
			}
			int sum = sumStartLign;
			for (int twNeg = 0; twNeg <= Math.min(twPos + 1, twNegMaxIndex); twNeg++) {
				if (twNeg > 0) {
					if (input[this.orderedAbsWeightsIndex[twNeg - 1]]) {
						sum--;
					}
				}
				// System.out.println(twPos + " " + twNeg + " " + sum);
				// int recomputedSum = 0;
				// for (int i = 0; i < twNeg; i++) {
				// if (input[this.orderedAbsWeightsIndex[i]]) {
				// recomputedSum--;
				// }
				// }
				// for (int i = twPos + 1; i < input.length; i++) {
				// if (input[this.orderedAbsWeightsIndex[i]]) {
				// recomputedSum++;
				// }
				// }
				// if (recomputedSum != sum) {
				// System.out.println("wtf");
				// }
				this.updateAgreement(twPos, twNeg, sum, outputProbs);
			}
		}
	}

	private void updateAgreement(int binarizationIndexPos, int binarizationIndexNeg, int sum,
			TernaryProbDistrib outputProbs) {
		TreeMap<Integer, TernaryProbDistrib> map = this.binarizationQuality
				.get(new TwPair(binarizationIndexPos, binarizationIndexNeg));
		synchronized (map) {
			TernaryProbDistrib dist = map.get(sum);
			if (dist == null) {
				dist = new TernaryProbDistrib();
				map.put(sum, dist);
				nbQualityCellsFilled.incrementAndGet();
			}
			dist.merge(outputProbs);
		}
	}

	public double getScore(double twPos, double twNeg, int th, int tl) {
		int twPosIndex = -1;
		for (int i = 0; i < this.orderedAbsWeightsIndex.length; i++) {
			double weight = this.realNeuron.getWeights()[this.orderedAbsWeightsIndex[i]];
			if (twPos < weight) {
				break;
			} else {
				twPosIndex++;
			}
		}
		int twNegIndex = 0;
		for (int i = 0; i < this.orderedAbsWeightsIndex.length; i++) {
			double weight = this.realNeuron.getWeights()[this.orderedAbsWeightsIndex[i]];
			if (twNeg <= weight) {
				break;
			} else {
				twNegIndex++;
			}
		}
		System.out.println(twPosIndex + " " + twNegIndex);
		return getScore(twPosIndex, twNegIndex, th, tl);
	}

	public double getScore(int twPosIndex, int twNegIndex, int th, int tl) {
		TreeMap<Integer, TernaryProbDistrib> map = this.binarizationQuality.get(new TwPair(twPosIndex, twNegIndex));
		TernaryConfusionMatrix currentDistrib = new TernaryConfusionMatrix();
		for (Entry<Integer, TernaryProbDistrib> en : map.entrySet()) {
			int sum = en.getKey();
			if (sum < tl) {
				currentDistrib.add(en.getValue(), 0);
			} else if (sum > th) {
				currentDistrib.add(en.getValue(), 2);
			} else {
				currentDistrib.add(en.getValue(), 1);
			}
		}
		return currentDistrib.getScore(this.scoreFun);
	}

	public TernarySolution findBestBinarizedConfiguration() {
		// weightBinarizationIndex is exclusive, so weightBinarizationIndex = 0
		// means nothing is binary

		return this.binarizationQuality.entrySet().parallelStream()
				.map(new Function<Entry<TwPair, TreeMap<Integer, TernaryProbDistrib>>, TernarySolution>() {

					@Override
					public TernarySolution apply(Entry<TwPair, TreeMap<Integer, TernaryProbDistrib>> t) {
						double currentBestScore = Double.NEGATIVE_INFINITY;
						TernarySolution currentBestParam = null;
						TreeMap<Integer, TernaryProbDistrib> map = t.getValue();
						TernaryConfusionMatrix lineEndDistrib = new TernaryConfusionMatrix();
						// with th max and tl min we always output 0
						for (TernaryProbDistrib distrib : map.values()) {
							if (distrib != null) {
								lineEndDistrib.add(distrib, 1);
							}
						}
						Iterator<Entry<Integer, TernaryProbDistrib>> thIterator = map.descendingMap().entrySet()
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
							// System.out.println("th=" + th + " possible tl are
							// " +
							// map.headMap(th + 1, true).keySet());
							Iterator<Entry<Integer, TernaryProbDistrib>> tlIterator = map.headMap(th + 1, true)
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
								double score = currentDistrib.getScore(scoreFun);
								if (currentBestParam == null || score > currentBestScore) {
									currentBestScore = score;
									currentBestParam = new TernarySolution(th, tl,
											t.getKey().twPos < 0
													? Math.nextDown(realNeuron.getWeights()[orderedAbsWeightsIndex[0]])
													: realNeuron.getWeights()[orderedAbsWeightsIndex[t.getKey().twPos]],
											t.getKey().twNeg == realNeuron.getWeights().length
													? Math.nextUp(
															realNeuron.getWeights()[orderedAbsWeightsIndex[realNeuron
																	.getWeights().length - 1]])
													: realNeuron.getWeights()[orderedAbsWeightsIndex[t.getKey().twNeg]],
											t.getKey().twPos, t.getKey().twNeg,
											new TernaryConfusionMatrix(currentDistrib), score);
								}
								// update currentDistrib
								// when th doesn'change but tl goes up some 0
								// answers
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
							// when th goes down and tl still min some 0 answers
							// become
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
						return currentBestParam;
					}
				}).max(new Comparator<TernarySolution>() {

					@Override
					public int compare(TernarySolution o1, TernarySolution o2) {
						return (int) Math.signum(o1.score - o2.score);
					}
				}).get();
	}

	private static class TwPair {
		public final int twPos;
		public final int twNeg;

		public TwPair(int twPos, int twNeg) {
			super();
			this.twPos = twPos;
			this.twNeg = twNeg;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + twNeg;
			result = prime * result + twPos;
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			TwPair other = (TwPair) obj;
			if (twNeg != other.twNeg)
				return false;
			if (twPos != other.twPos)
				return false;
			return true;
		}

	}

	public static void main(String[] args) throws Exception {
		String trainingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		// String outputFile = args[3];
		final int neuronIndex = 0;
		long start = System.currentTimeMillis();
		ASymBinarizer binarizer = new ASymBinarizer(new TanHNeuron(FilesProcessing.getWeights(weightsData, neuronIndex),
				FilesProcessing.getBias(biasData, neuronIndex), false), true, ScoreFunctions.AGREEMENT);
		int count = 0;
		for (boolean[] input : FilesProcessing.getTrainingSet(trainingData, 40)) {
			binarizer.update(input);
			count++;
			System.out.println(count);
		}
		System.out.println((System.currentTimeMillis() - start) / 1000 + " s");
		System.out.println(binarizer.findBestBinarizedConfiguration());
		// System.out.println(binarizer.getScore(981, 81, 2, 3));
		// System.out.println(binarizer.getScore(781, 437, -5, -4));
	}
}
