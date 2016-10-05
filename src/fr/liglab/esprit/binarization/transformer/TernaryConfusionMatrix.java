package fr.liglab.esprit.binarization.transformer;

import java.util.Arrays;

import fr.liglab.esprit.binarization.ScoreFunctions;
import fr.liglab.esprit.binarization.TernaryProbDistrib;

public class TernaryConfusionMatrix {
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
			line[i] += distrib.getProb(i);
		}
	}

	public void remove(TernaryProbDistrib distrib, int chosenOutput) {
		double[] line = this.matrix[chosenOutput];
		for (int i = 0; i < line.length; i++) {
			line[i] -= distrib.getProb(i);

			if (line[i] < 0.) {
				if (line[i] > -0.000000001) {
					line[i] = 0;
				}
			}
		}
	}

	public void add(TernaryProbDistrib distrib, int chosenOutput, double scalling) {
		double[] line = this.matrix[chosenOutput];
		for (int i = 0; i < line.length; i++) {
			line[i] += distrib.getProb(i) * scalling;
		}
	}

	public void remove(TernaryProbDistrib distrib, int chosenOutput, double scalling) {
		double[] line = this.matrix[chosenOutput];
		for (int i = 0; i < line.length; i++) {
			line[i] -= distrib.getProb(i) * scalling;

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
		return Arrays.toString(this.matrix[0]) + "\n" + Arrays.toString(this.matrix[1]) + "\n"
				+ Arrays.toString(this.matrix[2]);
	}

}