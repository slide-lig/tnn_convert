package fr.liglab.esprit.binarization.neuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class TernaryWeightsNeuron implements TernaryOutputNeuron {
	private final double[] weights;
	private int th;
	private int tl;

	public TernaryWeightsNeuron(double[] weights, int th, int tl) {
		super();
		this.weights = weights;
		this.th = th;
		this.tl = tl;
	}

	public TernaryWeightsNeuron(double[] weights, double twPos, double twNeg, int th, int tl) {
		super();
		this.weights = weights;
		this.th = th;
		this.tl = tl;
		for (int i = 0; i < weights.length; i++) {
			if (this.weights[i] > twPos) {
				this.weights[i] = 1;
			} else if (this.weights[i] < twNeg) {
				this.weights[i] = -1;
			} else {
				this.weights[i] = 0;
			}
		}
	}

	public TernaryWeightsNeuron(double[] weights, int nbPosWeights, int nbNegWeights, int th, int tl) {
		super();
		List<Integer> posWeightsIndex = new ArrayList<>(weights.length);
		List<Integer> negWeightsIndex = new ArrayList<>(weights.length);
		for (int i = 0; i < weights.length; i++) {
			if (weights[i] > 0) {
				posWeightsIndex.add(i);
			} else if (weights[i] < 0) {
				negWeightsIndex.add(i);
			}
		}
		if (posWeightsIndex.isEmpty() || negWeightsIndex.isEmpty()) {
			throw new RuntimeException("cannot force pos/neg tw if all weights are positive or negative");
		} else {
			Collections.sort(posWeightsIndex, new Comparator<Integer>() {

				@Override
				public int compare(Integer o1, Integer o2) {
					Double d1 = weights[o1];
					Double d2 = weights[o2];
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
			Arrays.fill(weights, 0.);
			Iterator<Integer> iter = posWeightsIndex.iterator();
			for (int i = 0; i < nbPosWeights; i++) {
				weights[iter.next()] = 1.;
			}
			iter = negWeightsIndex.iterator();
			for (int i = 0; i < nbNegWeights; i++) {
				weights[iter.next()] = -1.;
			}
			this.weights = weights;
			this.th = th;
			this.tl = tl;
		}
	}

	public int getSum(byte[] input) {
		int sum = 0;
		for (int i = 0; i < input.length; i++) {
			if (input[i] > 0) {
				if (this.weights[i] > 0.) {
					sum += input[i];
				} else if (this.weights[i] < 0.) {
					sum -= input[i];
				}
			} else if (input[i] < 0) {
				if (this.weights[i] > 0.) {
					sum -= input[i];
				} else if (this.weights[i] < 0.) {
					sum += input[i];
				}
			}
		}
		return sum;
	}

	public TernaryProbDistrib getOutputProbs(byte[] input) {
		double[] probs = new double[3];
		int sum = this.getSum(input);
		if (sum > this.th) {
			probs[0] = 0.;
			probs[1] = 0.;
			probs[2] = 1.;
		} else if (sum < this.tl) {
			probs[0] = 1.;
			probs[1] = 0.;
			probs[2] = 0.;
		} else {
			probs[0] = 0.;
			probs[1] = 1.;
			probs[2] = 0.;
		}
		return new TernaryProbDistrib(probs);
	}

	@Override
	public TernaryProbDistrib getConvOutputProbs(byte[] input, int startX, int startY, int dataXSize, int dataYSize,
			short convXSize, short convYSize, int nbChannels) {
		double[] probs = new double[3];
		int sum = 0;
		for (int i = 0; i < convXSize; i++) {
			if (startX + i >= 0 && startX + i < dataXSize) {
				for (int j = 0; j < convYSize; j++) {
					if (startY + j >= 0 && startY + j < dataYSize) {
						for (int channel = 0; channel < nbChannels; channel++) {
							final int convPos = j * convXSize + i + channel * convXSize * convYSize;
							final int pos = (j + startY) * dataXSize + (i + startX) + channel * dataXSize * dataYSize;
							if (input[pos] > 0) {
								if (this.weights[convPos] > 0.) {
									sum += input[pos];
								} else if (this.weights[convPos] < 0.) {
									sum -= input[pos];
								}
							} else if (input[pos] < 0) {
								if (this.weights[convPos] > 0.) {
									sum -= input[pos];
								} else if (this.weights[convPos] < 0.) {
									sum += input[pos];
								}
							}
						}
					}
				}
			}
		}
		if (sum > this.th) {
			probs[0] = 0.;
			probs[1] = 0.;
			probs[2] = 1.;
		} else if (sum < this.tl) {
			probs[0] = 1.;
			probs[1] = 0.;
			probs[2] = 0.;
		} else {
			probs[0] = 0.;
			probs[1] = 1.;
			probs[2] = 0.;
		}
		return new TernaryProbDistrib(probs);
	}

	@Override
	public final double[] getWeights() {
		return this.weights;
	}

	public final int getTh() {
		return th;
	}

	public final void setTh(int th) {
		this.th = th;
	}

	public final int getTl() {
		return tl;
	}

	public final void setTl(int tl) {
		this.tl = tl;
	}
}
