package fr.liglab.esprit.binarization.analysis;

import java.io.BufferedReader;
import java.io.FileReader;

public class BestScoreSearch {

	public static class ResultPosition {
		private final int x;
		private final int y;
		private final double s;

		public ResultPosition(int x, int y, double s) {
			super();
			this.x = x;
			this.y = y;
			this.s = s;
		}

		@Override
		public String toString() {
			return "ResultPosition [x=" + x + ", y=" + y + ", s=" + s + "]";
		}

	}

	private final double[][] scoreSpace;
	private int nbOptionsTested = 0;
	private int nbXCalls = 0;
	private int nbYCalls = 0;

	public BestScoreSearch(double[][] scoreSpace) {
		super();
		this.scoreSpace = scoreSpace;
	}

	public ResultPosition getActualBest() {
		ResultPosition best = null;
		for (int i = 0; i < this.scoreSpace.length; i++) {
			for (int j = 0; j < this.scoreSpace[i].length; j++) {
				if (best == null || this.scoreSpace[i][j] > best.s) {
					best = new ResultPosition(i, j, scoreSpace[i][j]);
				}
			}
		}
		return best;
	}

	public ResultPosition searchBestDichotomic() {
		return this.searchBestDichotomic(0, this.scoreSpace.length);
	}

	public final int getNbOptionsTested() {
		return nbOptionsTested;
	}

	public ResultPosition searchGrid(int deltaX, int deltaY) {
		ResultPosition bestPos = null;
		for (int i = deltaX; i < this.scoreSpace.length; i += deltaX) {
			for (int j = deltaY; j < this.scoreSpace[i].length; j += deltaY) {
				this.nbOptionsTested++;
				if (bestPos == null || this.scoreSpace[i][j] > bestPos.s) {
					bestPos = new ResultPosition(i, j, this.scoreSpace[i][j]);
				}
			}
		}
		return bestPos;
	}

	private ResultPosition searchBestDichotomic(int fromX, int toX) {
		nbXCalls++;
		// if (fromX == 278 && toX == 281) {
		// System.out.println("found it");
		// }
		// System.out.println("Starting X " + fromX + " - " + toX);
		if (toX - fromX == 1) {
			return searchBestDichotomic(fromX, 0, this.scoreSpace[fromX].length);
		} else if (toX - fromX == 2) {
			ResultPosition r1 = searchBestDichotomic(fromX, 0, this.scoreSpace[fromX].length);
			ResultPosition r2 = searchBestDichotomic(fromX + 1, 0, this.scoreSpace[fromX + 1].length);
			if (r1.s > r2.s) {
				return r1;
			} else {
				return r2;
			}
		} else {
			int length = toX - fromX;
			int segment1Size = (length - 2) / 3;
			int breakPoint1Pos = fromX + segment1Size + 1;
			ResultPosition lim1Score = this.searchBestDichotomic(breakPoint1Pos, 0,
					this.scoreSpace[breakPoint1Pos].length);
			int segment2Size = (length - segment1Size - 2) / 2;
			int breakPoint2Pos = breakPoint1Pos + segment2Size + 1;
			ResultPosition lim2Score = this.searchBestDichotomic(breakPoint2Pos, 0,
					this.scoreSpace[breakPoint2Pos].length);
			if (lim1Score.s > lim2Score.s) {
				return searchBestDichotomic(fromX, breakPoint2Pos);
			} else {
				return searchBestDichotomic(breakPoint1Pos + 1, toX);
			}
		}
	}

	private ResultPosition searchBestDichotomic(int x, int fromY, int toY) {
		nbYCalls++;
		// System.out.println("Y " + x + " - " + fromY + " - " + toY);
		if (toY - fromY == 1) {
			this.nbOptionsTested += 1;
			return new ResultPosition(x, fromY, this.scoreSpace[x][fromY]);
		} else if (toY - fromY == 2) {
			nbOptionsTested += 2;
			if (this.scoreSpace[x][fromY] > this.scoreSpace[x][fromY + 1]) {
				return new ResultPosition(x, fromY, this.scoreSpace[x][fromY]);
			} else {
				return new ResultPosition(x, fromY + 1, this.scoreSpace[x][fromY + 1]);
			}
		} else {
			nbOptionsTested += 2;
			int length = toY - fromY;
			int segment1Size = (length - 2) / 3;
			int breakPoint1Pos = fromY + segment1Size;
			ResultPosition lim1Score = new ResultPosition(x, breakPoint1Pos, this.scoreSpace[x][breakPoint1Pos]);
			int segment2Size = (length - segment1Size - 2) / 2;
			int breakPoint2Pos = breakPoint1Pos + segment2Size + 1;
			ResultPosition lim2Score = new ResultPosition(x, breakPoint2Pos, this.scoreSpace[x][breakPoint2Pos]);
			if (lim1Score.s > lim2Score.s) {
				return searchBestDichotomic(x, fromY, breakPoint2Pos);
			} else {
				return searchBestDichotomic(x, breakPoint1Pos + 1, toY);
			}
		}
	}

	public ResultPosition searchExhaustiveAround(final int x, final int y, final int dx, int dy) {
		ResultPosition bestPos = null;
		for (int i = Math.max(0, x - dx / 2); i < Math.min(this.scoreSpace.length, x + 1 + dx / 2); i++) {
			for (int j = Math.max(0, y - dy / 2); j < Math.min(this.scoreSpace.length, y + 1 + dy / 2); j++) {
				if (!(i == x && j == y)) {
					this.nbOptionsTested++;
				}
				if (bestPos == null || this.scoreSpace[i][j] > bestPos.s) {
					bestPos = new ResultPosition(i, j, this.scoreSpace[i][j]);
				}
			}
		}
		return bestPos;
	}

	public static void main(String[] args) throws Exception {
		// double[][] mat = new double[][] { { 0.37, 0.48, 0.43, 0.38, 0.32 }, {
		// 0.4, 0.46, 0.43, 0.42, 0.34 },
		// { 0.7, 0.75, 0.9, 0.85, 0.8 }, { 0.47, 0.72, 0.78, 0.83, 0.82 }, {
		// 0.41, 0.72, 0.67, 0.65, 0.61 } };
//		double[][] mat = new double[10000][10000];
//		for (int i = 0; i < mat.length; i++) {
//			for (int j = 0; j < mat[i].length; j++) {
//				mat[i][j] = Math.random();
//			}
//		}
		String file = "/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/binary_agreement_asym_distrib.txt_neuron_19-AGREEMENT";
		int maxX = 0;
		int minX = Integer.MAX_VALUE;
		int maxY = 0;
		int minY = Integer.MAX_VALUE;
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while ((line = br.readLine()) != null) {
			String[] sp = line.split(",");
			int x = Integer.parseInt(sp[2]);
			int y = Integer.parseInt(sp[3]);
			maxX = Math.max(maxX, x);
			minX = Math.min(minX, x);
			maxY = Math.max(maxY, y);
			minY = Math.min(minY, y);
		}
		br.close();
		double[][] mat = new double[maxX - minX + 1][maxY - minY + 1];
		br = new BufferedReader(new FileReader(file));
		while ((line = br.readLine()) != null) {
			String[] sp = line.split(",");
			int x = Integer.parseInt(sp[2]);
			int y = Integer.parseInt(sp[3]);
			double s = Double.parseDouble(sp[6]);
			mat[x - minX][y - minY] = s;
		}
		br.close();
		BestScoreSearch bss = new BestScoreSearch(mat);
		ResultPosition realBest = bss.getActualBest();
		System.out.println("real best " + realBest);
		ResultPosition dichotomicBest = bss.searchBestDichotomic();
		System.out.println("dichotomic " + dichotomicBest);
		System.out.println(bss.getNbOptionsTested() + " out of " + (mat.length * mat[0].length));
		// System.out.println(bss.nbXCalls + " - " + bss.nbYCalls);
		ResultPosition bestAround = bss.searchExhaustiveAround(dichotomicBest.x, dichotomicBest.y,
				2 * (int) Math.log(mat.length), 2 * (int) Math.log(mat[0].length));
		System.out.println("dichotomic + around " + bestAround);
		System.out.println(bss.getNbOptionsTested() + " out of " + (mat.length * mat[0].length));
		bss.nbOptionsTested = 0;
		ResultPosition bestGrid = bss.searchGrid((int) Math.sqrt(mat.length), (int) Math.sqrt(mat[0].length));
		System.out.println("grid " + bestGrid);
		System.out.println(bss.getNbOptionsTested() + " out of " + (mat.length * mat[0].length));
		ResultPosition bestAroundGrid = bss.searchExhaustiveAround(bestGrid.x, bestGrid.y,
				2 * (int) Math.sqrt(mat.length), 2 * (int) Math.sqrt(mat[0].length));
		System.out.println("grid + around " + bestAroundGrid);
		System.out.println(bss.getNbOptionsTested() + " out of " + (mat.length * mat[0].length));
		// System.out.println(Arrays.stream(mat[277]).max().getAsDouble());
		// System.out.println(Arrays.stream(mat[278]).max().getAsDouble());
		// System.out.println(Arrays.stream(mat[279]).max().getAsDouble());
		// System.out.println(Arrays.stream(mat[280]).max().getAsDouble());
	}

}
