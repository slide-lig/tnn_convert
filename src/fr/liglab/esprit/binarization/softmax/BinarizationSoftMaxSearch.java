package fr.liglab.esprit.binarization.softmax;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.stream.IntStream;

public class BinarizationSoftMaxSearch {

	private final CachedSoftmax scoreSpace;
	private int nbOptionsTested = 0;
	private final CachedSoftmax[] cachedNeurons;
	private final SoftMaxConfig[] existingConfigs;
	private final int[] groundTruth;
	private final int configuredNeuronIndex;

	public BinarizationSoftMaxSearch(CachedSoftmax[] cachedNeurons, SoftMaxConfig[] existingConfigs, int[] groundTruth,
			int configuredNeuronIndex) {
		super();
		this.scoreSpace = cachedNeurons[configuredNeuronIndex];
		this.cachedNeurons = cachedNeurons;
		this.existingConfigs = existingConfigs;
		this.groundTruth = groundTruth;
		this.configuredNeuronIndex = configuredNeuronIndex;
	}

	public SoftMaxConfig getActualBest() {
		SoftMaxConfig best = null;
		for (int i = 0; i < this.scoreSpace.getNbPosPossibilities(); i++) {
			for (int j = 0; j < this.scoreSpace.getNbNegPossibilities(); j++) {
				this.nbOptionsTested++;
				if (this.nbOptionsTested % 100 == 0) {
					System.out.println("done " + ((double) this.nbOptionsTested
							/ (this.scoreSpace.getNbPosPossibilities() * this.scoreSpace.getNbNegPossibilities())));
				}
				SoftMaxConfig pos = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
						this.groundTruth, this.configuredNeuronIndex, i, j);
				if (best == null || pos.getScore() > best.getScore()) {
					best = pos;
				}
			}
		}
		return best;
	}

	public SoftMaxConfig getActualBestParallel() {
		return IntStream.range(0, this.scoreSpace.getNbPosPossibilities()).parallel()
				.mapToObj(i -> IntStream.range(0, this.scoreSpace.getNbNegPossibilities()).parallel()
						.mapToObj(j -> CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
								this.groundTruth, this.configuredNeuronIndex, i, j))
						.max(SoftMaxConfig.comparator).get())
				.max(SoftMaxConfig.comparator).get();
	}

	@SuppressWarnings("unused")
	private SoftMaxConfig getActualBest(int x) {
		SoftMaxConfig best = null;
		for (int j = 0; j < this.scoreSpace.getNbNegPossibilities(); j++) {
			this.nbOptionsTested++;
			SoftMaxConfig pos = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs, this.groundTruth,
					this.configuredNeuronIndex, x, j);
			if (best == null || pos.getScore() > best.getScore()) {
				best = pos;
			}
		}
		return best;
	}

	public SoftMaxConfig getScoreHeatMap(String file) throws IOException {
		BufferedWriter bw = new BufferedWriter(new FileWriter(file));
		SoftMaxConfig best = null;
		for (int i = 0; i < this.scoreSpace.getNbPosPossibilities(); i++) {
			for (int j = 0; j < this.scoreSpace.getNbNegPossibilities(); j++) {
				this.nbOptionsTested++;
				if (this.nbOptionsTested % 100 == 0) {
					System.out.println("done " + ((double) this.nbOptionsTested
							/ (this.scoreSpace.getNbPosPossibilities() * this.scoreSpace.getNbNegPossibilities())));
				}
				SoftMaxConfig pos = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
						this.groundTruth, this.configuredNeuronIndex, i, j);
				bw.write(i + "\t" + j + "\t" + pos.getScore() + "\n");
				if (best == null || pos.getScore() > best.getScore()) {
					best = pos;
				}
			}
		}
		bw.close();
		return best;
	}

	public SoftMaxConfig searchBestLogLog() {
		SoftMaxConfig c1 = this.searchBestDichotomic();
		return this.searchExhaustiveAround(c1.getNbPosWeights(), c1.getNbNegWeights(),
				(int) Math.floor(2 * Math.log(this.scoreSpace.getNbPosPossibilities())) + 1,
				(int) Math.floor(2 * Math.log(this.scoreSpace.getNbNegPossibilities())) + 1);
	}

	public SoftMaxConfig searchBestSqrtSqrt() {
		int gridXDelta = (int) Math.floor(Math.sqrt(this.scoreSpace.getNbPosPossibilities()));
		int gridYDelta = (int) Math.floor(Math.sqrt(this.scoreSpace.getNbNegPossibilities()));
		SoftMaxConfig c1 = this.searchGrid(gridXDelta, gridYDelta);
		return this.searchExhaustiveAround(c1.getNbPosWeights(), c1.getNbNegWeights(), (gridXDelta + 1) * 2,
				(gridYDelta + 1) * 2);
	}

	public SoftMaxConfig searchBestDichotomic() {
		return this.searchBestDichotomic(0, this.scoreSpace.getNbPosPossibilities());
	}

	public final int getNbOptionsTested() {
		return nbOptionsTested;
	}

	public SoftMaxConfig searchGrid(int deltaX, int deltaY) {
		SoftMaxConfig bestPos = null;
		for (int i = deltaX; i < this.scoreSpace.getNbPosPossibilities(); i += deltaX) {
			for (int j = deltaY; j < this.scoreSpace.getNbNegPossibilities(); j += deltaY) {
				this.nbOptionsTested++;
				SoftMaxConfig pos = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
						this.groundTruth, this.configuredNeuronIndex, i, j);
				if (bestPos == null || pos.getScore() > bestPos.getScore()) {
					bestPos = pos;
				}
			}
		}
		return bestPos;
	}

	private SoftMaxConfig searchBestDichotomic(int fromX, int toX) {
		// if (fromX == 278 && toX == 281) {
		// System.out.println("found it");
		// }
		// System.out.println("Starting X " + fromX + " - " + toX);
		if (toX - fromX == 1) {
			return searchBestDichotomic(fromX, 0, this.scoreSpace.getNbNegPossibilities());
		} else if (toX - fromX == 2) {
			SoftMaxConfig r1 = searchBestDichotomic(fromX, 0, this.scoreSpace.getNbNegPossibilities());
			SoftMaxConfig r2 = searchBestDichotomic(fromX + 1, 0, this.scoreSpace.getNbNegPossibilities());
			if (r1.getScore() > r2.getScore()) {
				return r1;
			} else {
				return r2;
			}
		} else {
			int length = toX - fromX;
			int segment1Size = (length - 2) / 3;
			int breakPoint1Pos = fromX + segment1Size + 1;
			SoftMaxConfig lim1Score = this.searchBestDichotomic(breakPoint1Pos, 0,
					this.scoreSpace.getNbNegPossibilities());
			int segment2Size = (length - segment1Size - 2) / 2;
			int breakPoint2Pos = breakPoint1Pos + segment2Size + 1;
			SoftMaxConfig lim2Score = this.searchBestDichotomic(breakPoint2Pos, 0,
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

	private SoftMaxConfig searchBestDichotomic(int x, int fromY, int toY) {
		// System.out.println("Y " + x + " - " + fromY + " - " + toY);
		if (toY - fromY == 1) {
			this.nbOptionsTested += 1;
			return CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs, this.groundTruth,
					this.configuredNeuronIndex, x, fromY);
		} else if (toY - fromY == 2) {
			nbOptionsTested += 2;
			SoftMaxConfig first = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
					this.groundTruth, this.configuredNeuronIndex, x, fromY);
			SoftMaxConfig second = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
					this.groundTruth, this.configuredNeuronIndex, x, fromY + 1);
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
			SoftMaxConfig lim1Score = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
					this.groundTruth, this.configuredNeuronIndex, x, breakPoint1Pos);
			int segment2Size = (length - segment1Size - 2) / 2;
			int breakPoint2Pos = breakPoint1Pos + segment2Size + 1;
			SoftMaxConfig lim2Score = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
					this.groundTruth, this.configuredNeuronIndex, x, breakPoint2Pos);
			// System.out.println(lim1Score + "\t" + lim2Score);
			if (lim1Score.getScore() > lim2Score.getScore()) {
				return searchBestDichotomic(x, fromY, breakPoint2Pos);
			} else {
				return searchBestDichotomic(x, breakPoint1Pos + 1, toY);
			}
		}
	}

	public SoftMaxConfig searchExhaustiveAround(final int x, final int y, final int dx, int dy) {
		SoftMaxConfig bestPos = null;
		for (int i = Math.max(0, x - dx / 2); i < Math.min(this.scoreSpace.getNbPosPossibilities(),
				x + 1 + dx / 2); i++) {
			for (int j = Math.max(0, y - dy / 2); j < Math.min(this.scoreSpace.getNbNegPossibilities(),
					y + 1 + dy / 2); j++) {
				this.nbOptionsTested++;
				SoftMaxConfig pos = CachedSoftmax.getBestConfig(this.cachedNeurons, this.existingConfigs,
						this.groundTruth, this.configuredNeuronIndex, i, j);
				if (bestPos == null || pos.getScore() > bestPos.getScore()) {
					bestPos = pos;
				}
			}
		}
		return bestPos;
	}

}
