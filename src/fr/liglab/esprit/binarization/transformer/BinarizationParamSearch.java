package fr.liglab.esprit.binarization.transformer;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.stream.IntStream;

import fr.liglab.esprit.binarization.FilesProcessing;
import fr.liglab.esprit.binarization.neuron.CachedBinarization;
import fr.liglab.esprit.binarization.neuron.TanHNeuron;

public class BinarizationParamSearch {

	private final CachedBinarization scoreSpace;
	private int nbOptionsTested = 0;

	public BinarizationParamSearch(CachedBinarization scoreSpace) {
		super();
		this.scoreSpace = scoreSpace;
	}

	public TernaryConfig getActualBest() {
		TernaryConfig best = null;
		for (int i = 0; i < this.scoreSpace.getNbPosPossibilities(); i++) {
			for (int j = 0; j < this.scoreSpace.getNbNegPossibilities(); j++) {
				this.nbOptionsTested++;
				if (this.nbOptionsTested % 100 == 0) {
					System.out.println("done " + ((double) this.nbOptionsTested
							/ (this.scoreSpace.getNbPosPossibilities() * this.scoreSpace.getNbNegPossibilities())));
				}
				TernaryConfig pos = this.scoreSpace.getBestConfig(i, j);
				if (best == null || pos.getScore() > best.getScore()) {
					best = pos;
				}
			}
		}
		return best;
	}

	public TernaryConfig getActualBestParallel() {
		return IntStream.range(0, this.scoreSpace.getNbPosPossibilities()).parallel()
				.mapToObj(i -> IntStream.range(0, this.scoreSpace.getNbNegPossibilities()).parallel()
						.mapToObj(j -> this.scoreSpace.getBestConfig(i, j)).max(TernaryConfig.comparator).get())
				.max(TernaryConfig.comparator).get();
	}

	@SuppressWarnings("unused")
	private TernaryConfig getActualBest(int x) {
		TernaryConfig best = null;
		for (int j = 0; j < this.scoreSpace.getNbNegPossibilities(); j++) {
			this.nbOptionsTested++;
			TernaryConfig pos = this.scoreSpace.getBestConfig(x, j);
			if (best == null || pos.getScore() > best.getScore()) {
				best = pos;
			}
		}
		return best;
	}

	public TernaryConfig getScoreHeatMap(String file) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(file));
		TernaryConfig best = null;
		for (int i = 0; i < this.scoreSpace.getNbPosPossibilities(); i++) {
			for (int j = 0; j < this.scoreSpace.getNbNegPossibilities(); j++) {
				this.nbOptionsTested++;
				if (this.nbOptionsTested % 100 == 0) {
					System.out.println("done " + ((double) this.nbOptionsTested
							/ (this.scoreSpace.getNbPosPossibilities() * this.scoreSpace.getNbNegPossibilities())));
				}
				TernaryConfig pos = this.scoreSpace.getBestConfig(i, j);
				bw.write(i + "\t" + j + "\t" + pos.getScore() + "\n");
				if (best == null || pos.getScore() > best.getScore()) {
					best = pos;
				}
			}
		}
		bw.close();
		return best;
	}

	public TernaryConfig searchBestLogLog() {
		TernaryConfig c1 = this.searchBestDichotomic();
		return this.searchExhaustiveAround(c1.getNbPosWeights(), c1.getNbNegWeights(),
				(int) Math.floor(2 * Math.log(this.scoreSpace.getNbPosPossibilities())) + 1,
				(int) Math.floor(2 * Math.log(this.scoreSpace.getNbNegPossibilities())) + 1);
	}

	public TernaryConfig searchBestSqrtSqrt() {
		int gridXDelta = (int) Math.floor(Math.sqrt(this.scoreSpace.getNbPosPossibilities()));
		int gridYDelta = (int) Math.floor(Math.sqrt(this.scoreSpace.getNbNegPossibilities()));
		TernaryConfig c1 = this.searchGrid(gridXDelta, gridYDelta);
		return this.searchExhaustiveAround(c1.getNbPosWeights(), c1.getNbNegWeights(), (gridXDelta + 1) * 2,
				(gridYDelta + 1) * 2);
	}

	public TernaryConfig searchBestDichotomic() {
		return this.searchBestDichotomic(0, this.scoreSpace.getNbPosPossibilities());
	}

	public final int getNbOptionsTested() {
		return nbOptionsTested;
	}

	public TernaryConfig searchGrid(int deltaX, int deltaY) {
		TernaryConfig bestPos = null;
		for (int i = deltaX; i < this.scoreSpace.getNbPosPossibilities(); i += deltaX) {
			for (int j = deltaY; j < this.scoreSpace.getNbNegPossibilities(); j += deltaY) {
				this.nbOptionsTested++;
				TernaryConfig pos = this.scoreSpace.getBestConfig(i, j);
				if (bestPos == null || pos.getScore() > bestPos.getScore()) {
					bestPos = pos;
				}
			}
		}
		return bestPos;
	}

	private TernaryConfig searchBestDichotomic(int fromX, int toX) {
		// if (fromX == 278 && toX == 281) {
		// System.out.println("found it");
		// }
		// System.out.println("Starting X " + fromX + " - " + toX);
		if (toX - fromX == 1) {
			return searchBestDichotomic(fromX, 0, this.scoreSpace.getNbNegPossibilities());
		} else if (toX - fromX == 2) {
			TernaryConfig r1 = searchBestDichotomic(fromX, 0, this.scoreSpace.getNbNegPossibilities());
			TernaryConfig r2 = searchBestDichotomic(fromX + 1, 0, this.scoreSpace.getNbNegPossibilities());
			if (r1.getScore() > r2.getScore()) {
				return r1;
			} else {
				return r2;
			}
		} else {
			int length = toX - fromX;
			int segment1Size = (length - 2) / 3;
			int breakPoint1Pos = fromX + segment1Size + 1;
			TernaryConfig lim1Score = this.searchBestDichotomic(breakPoint1Pos, 0,
					this.scoreSpace.getNbNegPossibilities());
			int segment2Size = (length - segment1Size - 2) / 2;
			int breakPoint2Pos = breakPoint1Pos + segment2Size + 1;
			TernaryConfig lim2Score = this.searchBestDichotomic(breakPoint2Pos, 0,
					this.scoreSpace.getNbNegPossibilities());
			// System.out.println(lim1Score + " (" +
			// getActualBest(breakPoint1Pos) + ")" + "\t" + lim2Score + " ("
			// + getActualBest(breakPoint2Pos) + ")");
			if (lim1Score.getScore() > lim2Score.getScore()) {
				return searchBestDichotomic(fromX, breakPoint2Pos);
			} else {
				return searchBestDichotomic(breakPoint1Pos + 1, toX);
			}
		}
	}

	private TernaryConfig searchBestDichotomic(int x, int fromY, int toY) {
		// System.out.println("Y " + x + " - " + fromY + " - " + toY);
		if (toY - fromY == 1) {
			this.nbOptionsTested += 1;
			return this.scoreSpace.getBestConfig(x, fromY);
		} else if (toY - fromY == 2) {
			nbOptionsTested += 2;
			TernaryConfig first = this.scoreSpace.getBestConfig(x, fromY);
			TernaryConfig second = this.scoreSpace.getBestConfig(x, fromY + 1);
			if (first.getScore() > second.getScore()) {
				return first;
			} else {
				return second;
			}
		} else {
			nbOptionsTested += 2;
			int length = toY - fromY;
			int segment1Size = (length - 2) / 3;
			int breakPoint1Pos = fromY + segment1Size;
			TernaryConfig lim1Score = this.scoreSpace.getBestConfig(x, breakPoint1Pos);
			int segment2Size = (length - segment1Size - 2) / 2;
			int breakPoint2Pos = breakPoint1Pos + segment2Size + 1;
			TernaryConfig lim2Score = this.scoreSpace.getBestConfig(x, breakPoint2Pos);
			// System.out.println(lim1Score + "\t" + lim2Score);
			if (lim1Score.getScore() > lim2Score.getScore()) {
				return searchBestDichotomic(x, fromY, breakPoint2Pos);
			} else {
				return searchBestDichotomic(x, breakPoint1Pos + 1, toY);
			}
		}
	}

	public TernaryConfig searchExhaustiveAround(final int x, final int y, final int dx, int dy) {
		TernaryConfig bestPos = null;
		for (int i = Math.max(0, x - dx / 2); i < Math.min(this.scoreSpace.getNbPosPossibilities(),
				x + 1 + dx / 2); i++) {
			for (int j = Math.max(0, y - dy / 2); j < Math.min(this.scoreSpace.getNbNegPossibilities(),
					y + 1 + dy / 2); j++) {
				this.nbOptionsTested++;
				TernaryConfig pos = this.scoreSpace.getBestConfig(i, j);
				if (bestPos == null || pos.getScore() > bestPos.getScore()) {
					bestPos = pos;
				}
			}
		}
		return bestPos;
	}

	public static void main(String[] args) throws Exception {
		double[] weights = FilesProcessing.getWeights("/Users/vleroy/Desktop/neuron26.txt", 0);
		double bias = FilesProcessing.getBias("/Users/vleroy/Desktop/bias26.txt", 0);
		TanHNeuron nOrigin = new TanHNeuron(weights, bias, false);
		List<byte[]> input = FilesProcessing.getAllTrainingSet(
				"/Users/vleroy/workspace/esprit/mnist_binary/MNIST_32_32/dataTrain.txt", Integer.MAX_VALUE);
		CachedBinarization cb = new CachedBinarization(nOrigin, input, null);
		// TernaryWeightsNeuron nBinarized = new
		// TernaryWeightsNeuron(Arrays.copyOf(weights, weights.length),
		// 0.030318,
		// -0.029748, 9, -9);
		// int nbPosWeights = nBinarized.getNbPosWeights();
		// int nbNegWeights = nBinarized.getNbNegWeights();
		// System.out.println(cb.getBestConfig(114, 71));
		BinarizationParamSearch bss = new BinarizationParamSearch(cb);
		// System.out.println(bss.getActualBest());
		long start = System.currentTimeMillis();
		System.out.println(bss.searchBestLogLog() + " (best achievable " + nOrigin.getMaxAgreement() + ")");
		long end = System.currentTimeMillis();
		System.out.println("Runtime = " + (end - start) + " ms");

	}

}
