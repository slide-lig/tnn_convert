package fr.liglab.esprit.binarization;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class FilesProcessing {
	// private static final int nbAlwaysZeroPixels = 348;
	// private static final boolean[] alwaysZeroPixels = { true, true, true,
	// true, true, true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, false, false,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, false, true, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, true, true, true,
	// true, true, true, true, true,
	// false, true, true, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, true,
	// true, true, true, true, true, true,
	// true, false, true, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, true, true, true, true, true,
	// true, true, true, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, true, true, true, true, true,
	// true, true, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, true, true, true,
	// true, true, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, true,
	// true, true, true, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// true, true, true, true, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, true, true, true, true, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, true, true, true, true, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, true, true, true, true, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, true, true, true, true, true, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, true, true, true, true, true, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, true, true, true, true, true, true,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, true, true, true, true,
	// true, true, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, true, true, true,
	// true, true, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, true, true, true,
	// true, true, true, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// true, true, true, true, true, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, true, true, true, true, true,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, true, true, true, true, true,
	// true, false, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, true, true, true, true,
	// true, true, true, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, true, true, true, true,
	// true, true, true, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, false, true, true, true, true,
	// true, true, true, false, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// false, true, true, true, true, true,
	// true, true, true, true, false, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, false,
	// true, true, true, true, true, true,
	// true, true, true, true, true, false, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, false, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, true, false, false, false, false, false,
	// false, false, false, false, false,
	// false, false, false, false, false, false, false, false, true, true, true,
	// true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true, true, true, true,
	// true, true, true, true, true, true,
	// true, true, true, true, true, true, true, true, true };

	public static int getHeadingZeros(String file) throws IOException {
		int nbZeros = Integer.MAX_VALUE;
		for (byte[] input : getAllTrainingSet(file, Integer.MAX_VALUE)) {
			int nb = 0;
			for (int i = 0; i < input.length && nb < nbZeros; i++) {
				if (input[i] == 0) {
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
		for (byte[] input : getAllTrainingSet(file, Integer.MAX_VALUE)) {
			int nb = 0;
			for (int i = input.length - 1; i >= 0 && nb < nbZeros; i--) {
				if (input[i] == 0) {
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
		for (byte[] input : getAllTrainingSet(file, Integer.MAX_VALUE)) {
			for (int i = 0; i < input.length; i++) {
				if (alwaysZero[i] && input[i] != 0) {
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
		for (byte[] input : getAllTrainingSet(file, Integer.MAX_VALUE)) {
			for (int i = 0; i < input.length; i++) {
				if (alwaysZero[i] && input[i] != 0) {
					alwaysZero[i] = false;
				}
			}
		}
		return alwaysZero;
	}

	public static int[] getDensity(String file) throws IOException {
		int[] histo = new int[1024];
		for (byte[] input : getAllTrainingSet(file, Integer.MAX_VALUE)) {
			int nb = 0;
			for (int i = 0; i < input.length; i++) {
				if (input[i] != 0) {
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

	// public static List<double[]> getFilteredWeights(String file, int
	// nbNeurons) throws IOException {
	// List<double[]> allWeights = new ArrayList<>();
	// BufferedReader br = new BufferedReader(new FileReader(file));
	// String line;
	//
	// while ((line = br.readLine()) != null) {
	// if (allWeights.size() == nbNeurons) {
	// break;
	// }
	// String[] input = line.split(",");
	// double[] weights = new double[input.length - nbAlwaysZeroPixels];
	// int insertPos = 0;
	// for (int i = 0; i < input.length; i++) {
	// if (!alwaysZeroPixels[i]) {
	// weights[insertPos] = Double.parseDouble(input[i]);
	// insertPos++;
	// }
	// }
	// allWeights.add(weights);
	// }
	// br.close();
	// return allWeights;
	// }

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

	// public static double[] getFilteredWeightsSingle(String file, int
	// neuronIndex) throws IOException {
	// // load neuron weights
	// BufferedReader br = new BufferedReader(new FileReader(file));
	// String line;
	// int lineNumber = 0;
	// double[] weights = null;
	// while ((line = br.readLine()) != null) {
	// if (lineNumber == neuronIndex) {
	// String[] input = line.split(",");
	// weights = new double[input.length - nbAlwaysZeroPixels];
	// int insertPos = 0;
	// for (int i = 0; i < input.length; i++) {
	// if (!alwaysZeroPixels[i]) {
	// weights[insertPos] = Double.parseDouble(input[i]);
	// insertPos++;
	// }
	// }
	// break;
	// }
	// lineNumber++;
	// }
	// br.close();
	// return weights;
	// }

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

	public static List<byte[]> getAllTrainingSet(String file, int nbSamples) throws IOException {
		List<byte[]> trainingset = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		int nbSamp = 0;
		while ((line = br.readLine()) != null) {
			if (nbSamp == nbSamples) {
				break;
			}
			if (!line.isEmpty()) {
				String[] data = line.split(",");
				byte[] input = new byte[data.length];
				for (int i = 0; i < data.length; i++) {
					input[i] = Byte.parseByte(data[i]);
				}
				trainingset.add(input);
			}
			nbSamp++;
		}
		br.close();
		return trainingset;
	}

	// public static List<byte[]> getFilteredTrainingSet(String file, int
	// nbSamples) throws IOException {
	// List<byte[]> trainingset = new ArrayList<>();
	// BufferedReader br = new BufferedReader(new FileReader(file));
	// String line;
	// int nbSamp = 0;
	// while ((line = br.readLine()) != null) {
	// if (nbSamp == nbSamples) {
	// break;
	// }
	// if (!line.isEmpty()) {
	// String[] data = line.split(",");
	// byte[] input = new byte[data.length - nbAlwaysZeroPixels];
	// int insertPos = 0;
	// for (int i = 0; i < data.length; i++) {
	// if (!alwaysZeroPixels[i]) {
	// input[insertPos] = Byte.parseByte(data[i]);
	// insertPos++;
	// }
	// }
	// trainingset.add(input);
	// }
	// nbSamp++;
	// }
	// br.close();
	// return trainingset;
	// }

	public static int[] getGroundTruth(String file, int nbSamples) throws IOException {
		List<String> trainingset = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		int nbSamp = 0;
		while ((line = br.readLine()) != null) {
			if (nbSamp == nbSamples) {
				break;
			}
			if (!line.isEmpty()) {
				trainingset.add(line);
			}
			nbSamp++;
		}
		br.close();
		int[] output = new int[trainingset.size()];
		for (int i = 0; i < trainingset.size(); i++) {
			output[i] = Integer.parseInt(trainingset.get(i)) - 1;
		}
		return output;
	}

	public static void genConvolutionInput(final int inputSize, final int convolutionSize, final String inputFile,
			final String outputFile) throws Exception {
		List<byte[]> trainingSet = FilesProcessing.getAllTrainingSet(inputFile, Integer.MAX_VALUE);
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
		for (byte[] input : trainingSet) {
			for (int x = 0; x < (inputSize - convolutionSize + 1); x++) {
				for (int y = 0; y < (inputSize - convolutionSize + 1); y++) {
					for (int i = 0; i < convolutionSize; i++) {
						for (int j = 0; j < convolutionSize; j++) {
							int pos = (i + x) * inputSize + (j + y);
							// System.out.println(x + "," + y + "," + i + "," +
							// j + "," + pos);
							if (!(i == 0 && j == 0)) {
								bw.write(",");
							}
							bw.write(Byte.toString(input[pos]));
						}
					}
					bw.write("\n");
				}
			}
		}
		bw.close();
	}

	public static void main(String[] args) throws Exception {
		String dataFile = "/Users/vleroy/Desktop/secret_binary_numbers_pi";
		float[] numbers = getActivationsBinary(dataFile);
		System.out.println(Arrays.toString(numbers));
		// int[] histo = getDensity(dataFile);
		// for (int i = 0; i < histo.length; i++) {
		// if (histo[i] > 0) {
		// System.out.println(i + "\t" + histo[i]);
		// }
		// }
		// System.out.println(getNbAlwaysZero(dataFile));
	}

	public static List<float[]> getAllActivations(String file, int nbNeurons) throws IOException {
		List<float[]> allActivations = new ArrayList<>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;

		while ((line = br.readLine()) != null) {
			if (allActivations.size() == nbNeurons) {
				break;
			}
			String[] input = line.split(",");
			float[] weights = new float[input.length];
			for (int i = 0; i < input.length; i++) {
				weights[i] = Float.parseFloat(input[i]);
			}
			allActivations.add(weights);
		}
		br.close();
		return allActivations;
	}

	public static float[] getActivations(String file) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(file));
		String line = br.readLine();
		br.close();
		String[] input = line.split(",");
		float[] weights = new float[input.length];
		for (int i = 0; i < input.length; i++) {
			weights[i] = Float.parseFloat(input[i]);
		}
		return weights;
	}

	public static float[] getActivationsBinary(String file) throws IOException {
		Path path = Paths.get(file);
		byte[] rawBytes = Files.readAllBytes(path);
		float[] weights = new float[rawBytes.length / 4];
		FloatBuffer fb = ByteBuffer.wrap(rawBytes).asFloatBuffer();
		fb.get(weights);
		return weights;
	}
}
