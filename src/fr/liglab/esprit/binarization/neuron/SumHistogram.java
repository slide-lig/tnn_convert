package fr.liglab.esprit.binarization.neuron;

public class SumHistogram {
	private final double[] dist;
	private final int offset;

	// both included
	public SumHistogram(int minVal, int maxVal) {
		this.offset = -minVal;
		this.dist = new double[maxVal - minVal + 1];
	}

	public void addOccurence(int sum, double occ) {
		this.dist[sum + offset] += occ;
	}

	public int findCrossPoint(SumHistogram other) {
		return this.findCrossPoint(other, -offset);
	}

	// this supposed to be left of other
	public int findCrossPoint(SumHistogram other, int from) {
		from += offset;
		int bestPos = -1;
		double bestScore = 0.;
		boolean potentialSwitch = true;
		for (int pos = from; pos < this.dist.length; pos++) {
			if (this.dist[pos] < other.dist[pos]) {
				if (potentialSwitch) {
					double score = (this.dist[pos] + other.dist[pos]) / 2;
					if (score > bestScore) {
						bestScore = score;
						bestPos = pos;
					}
				}
				potentialSwitch = false;
			} else {
				potentialSwitch = true;
			}
		}
		if (bestPos == -1) {
			return Integer.MAX_VALUE;
		} else {
			return bestPos - offset;
		}
	}

	public double getSum() {
		double s = 0;
		for (double d : this.dist) {
			s += d;
		}
		return s;
	}

	// both inclusive
	public double getSum(int from, int to) {
		from += offset;
		to += offset;
		double s = 0;
		for (int i = from; i <= to; i++) {
			s += this.dist[i];
		}
		return s;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < dist.length; i++) {
			if (dist[i] != 0) {
				sb.append((i - offset) + ": " + dist[i] + ", ");
			}
		}
		return "SumHistogram [dist=" + sb.toString() + ", offset=" + offset + ", sum=" + this.getSum() + "]";
	}

}
