package fr.liglab.esprit.binarization;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class FilesProcessing {
	private static final int nbAlwaysZeroPixels = 348;
	private static final boolean[] alwaysZeroPixels = { true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, true, true, true, true, true, true, true, true, true, true,
			true, true, false, true, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true,
			false, true, true, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true,
			true, false, true, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true,
			true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true,
			true, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, true, true, true,
			true, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, true,
			true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, true, true, true, true, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, true, true, true, true, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, true, true, true, true, true, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, true, true, true, true, true, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, true, true, true, true, true, true, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, true, true, true, true, true, true, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, true, true, true, true, true, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, true, true, true, true, true, true, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, false,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true,
			false, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true,
			true, false, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true,
			true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true,
			true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, false, true, true, true, true,
			true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, false, true, true, true, true, true,
			true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, false, true, true, true, true, true, true,
			true, true, true, true, true, false, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true,
			true, true, true, true, true, true, false, false, false, false, false, false, false, false, false, false,
			false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
			true, true, true, true, true, true, true, true, true };

	public static int getHeadingZeros(String file) throws IOException {
		int nbZeros = Integer.MAX_VALUE;
		for (boolean[] input : getTrainingSet(file, Integer.MAX_VALUE)) {
			int nb = 0;
			for (int i = 0; i < input.length && nb < nbZeros; i++) {
				if (!input[i]) {
					nb++;
				} else {
					break;
				}
			}
			nbZeros = nb;
		}
		return nbZeros;
	}

	public static int getTrailingZeros(String file) throws IOException {
		int nbZeros = Integer.MAX_VALUE;
		for (boolean[] input : getTrainingSet(file, Integer.MAX_VALUE)) {
			int nb = 0;
			for (int i = input.length - 1; i >= 0 && nb < nbZeros; i--) {
				if (!input[i]) {
					nb++;
				} else {
					break;
				}
			}
			nbZeros = nb;
		}
		return nbZeros;
	}

	public static int getNbAlwaysZero(String file) throws IOException {
		boolean[] alwaysZero = new boolean[1024];
		Arrays.fill(alwaysZero, true);
		for (boolean[] input : getTrainingSet(file, Integer.MAX_VALUE)) {
			for (int i = 0; i < input.length; i++) {
				if (alwaysZero[i] && input[i]) {
					alwaysZero[i] = false;
				}
			}
		}
		int count = 0;
		for (boolean b : alwaysZero) {
			if (b) {
				count++;
			}
		}
		return count;
	}

	public static boolean[] getAlwaysZero(String file) throws IOException {
		boolean[] alwaysZero = new boolean[1024];
		Arrays.fill(alwaysZero, true);
		for (boolean[] input : getTrainingSet(file, Integer.MAX_VALUE)) {
			for (int i = 0; i < input.length; i++) {
				if (alwaysZero[i] && input[i]) {
					alwaysZero[i] = false;
				}
			}
		}
		return alwaysZero;
	}

	public static int[] getDensity(String file) throws IOException {
		int[] histo = new int[1024];
		for (boolean[] input : getTrainingSet(file, Integer.MAX_VALUE)) {
			int nb = 0;
			for (int i = 0; i < input.length; i++) {
				if (input[i]) {
					nb++;
				}
			}
			histo[nb]++;
		}
		return histo;
	}

	public static double getCentileAbsWeight(String file, double sparsityConstraint) throws IOException {
		List<Double> l = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;

		while ((line = br.readLine()) != null) {
			String[] input = line.split(",");
			for (int i = 0; i < input.length; i++) {
				l.add(Math.abs(Double.parseDouble(input[i])));
			}
		}
		br.close();
		Collections.sort(l);
		return l.get((int) (sparsityConstraint * l.size()));
	}

	public static List<double[]> getAllWeights(String file, int nbNeurons) throws IOException {
		List<double[]> allWeights = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;

		while ((line = br.readLine()) != null) {
			if (allWeights.size() == nbNeurons) {
				break;
			}
			String[] input = line.split(",");
			double[] weights = new double[input.length];
			for (int i = 0; i < input.length; i++) {
				weights[i] = Double.parseDouble(input[i]);
			}
			allWeights.add(weights);
		}
		br.close();
		return allWeights;
	}

	public static double[] getWeights(String file, int neuronIndex) throws IOException {
		// load neuron weights
		BufferedReader br = new BufferedReader(new FileReader(file));
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
		return weights;
	}

	public static List<Double> getAllBias(String file, int nbNeurons) throws IOException {
		List<Double> allBias = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		while ((line = br.readLine()) != null) {
			if (allBias.size() == nbNeurons) {
				break;
			}

			allBias.add(Double.valueOf(line));
		}
		br.close();
		return allBias;
	}

	public static double getBias(String file, int neuronIndex) throws IOException {
		double bias = 0.;
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		int lineNumber = 0;
		while ((line = br.readLine()) != null) {
			if (lineNumber == neuronIndex) {
				bias = Double.valueOf(line);
				break;
			}
			lineNumber++;
		}
		br.close();
		return bias;
	}

	public static List<boolean[]> getTrainingSet(String file, int nbSamples) throws IOException {
		List<boolean[]> trainingset = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		int nbSamp = 0;
		while ((line = br.readLine()) != null) {
			if (nbSamp == nbSamples) {
				break;
			}
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
				trainingset.add(input);
			}
			nbSamp++;
		}
		br.close();
		return trainingset;
	}

	public static void main(String[] args) throws Exception {
		String dataFile = "/Users/vleroy/workspace/esprit/mnist_binary/MNIST_32_32/dataTrain.txt";
		// int[] histo = getDensity(dataFile);
		// for (int i = 0; i < histo.length; i++) {
		// if (histo[i] > 0) {
		// System.out.println(i + "\t" + histo[i]);
		// }
		// }
		System.out.println(getNbAlwaysZero(dataFile));
	}
}
