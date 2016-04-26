package fr.liglab.esprit.binarization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import fr.liglab.esprit.binarization.neuron.CachedBinarization;
import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.transformer.BinarizationParamSearch;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public class BinarizeAll {
	public static class RealNeuron {
		private double bias;
		private double[] weights;
		private int id;
	}

	private static final double DEFAULT_EXHAUSTIVE_THRESHOLD = 0.95;

	public static void main(String[] args) throws Exception {
		final Options options = new Options();
		options.addOption(Option.builder("t").longOpt("training").desc("Input training set").hasArg().argName("FILE")
				.required().build());
		options.addOption(Option.builder("r").longOpt("reference training")
				.desc("Input training set for the original neuron, to avoid propagating binarization errors").hasArg()
				.argName("FILE").required(false).build());
		options.addOption(Option.builder("w").longOpt("weights").desc("Original weights").hasArg().argName("FILE")
				.required().build());
		options.addOption(
				Option.builder("b").longOpt("bias").desc("Original bias").hasArg().argName("FILE").required().build());
		options.addOption(Option.builder("o").longOpt("output").desc("Neuron configuration output file").hasArg()
				.argName("FILE").required().build());
		options.addOption(Option.builder("e").longOpt("exhaustive")
				.desc("Threshold to go exhaustive (default " + DEFAULT_EXHAUSTIVE_THRESHOLD + ")").hasArg()
				.argName("THRESHOLD").build());
		options.addOption(Option.builder("d").longOpt("consider original neuron deterministic").hasArg(false).build());
		final CommandLineParser parser = new DefaultParser();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("BinarizeAll", options, true);
			System.exit(-1);
		}
		final String trainingData = cmd.getOptionValue("t");
		final String referenceTrainingData = cmd.getOptionValue("r", null);
		final String weightsData = cmd.getOptionValue("w");
		final String biasData = cmd.getOptionValue("b");
		final String outputFile = cmd.getOptionValue("o");
		final double exhaustiveThreshold = Double
				.parseDouble(cmd.getOptionValue("e", Double.toString(DEFAULT_EXHAUSTIVE_THRESHOLD)));
		final boolean deterministic = cmd.hasOption("d");
		if (exhaustiveThreshold < 0. || exhaustiveThreshold > 1.) {
			throw new RuntimeException("exhaustive threshold must be in [0,1]");
		}
		// double globalTw = FilesProcessing.getCentileAbsWeight(weightsData,
		// 0.80);
		final List<RealNeuron> lNeurons = new ArrayList<>();
		final List<double[]> allWeights = FilesProcessing.getAllWeights(weightsData, Integer.MAX_VALUE);
		final List<Double> allBias = FilesProcessing.getAllBias(biasData, Integer.MAX_VALUE);
		for (int i = 0; i < allWeights.size(); i++) {
			RealNeuron rl = new RealNeuron();
			rl.weights = allWeights.get(i);
			rl.bias = allBias.get(i);
			rl.id = i;
			lNeurons.add(rl);
		}
		final List<byte[]> images = FilesProcessing.getAllTrainingSet(trainingData, Integer.MAX_VALUE);
		final List<byte[]> referenceImages = (referenceTrainingData != null)
				? FilesProcessing.getAllTrainingSet(referenceTrainingData, Integer.MAX_VALUE) : null;
		final TernaryConfig[] solutions = new TernaryConfig[lNeurons.size()];
		final AtomicInteger nbDone = new AtomicInteger();
		final List<RealNeuron> neuronRerun = new ArrayList<>();
		lNeurons.parallelStream().forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(final RealNeuron t) {
				final TanHNeuron originalNeuron = new TanHNeuron(t.weights, t.bias, deterministic);
				final BinarizationParamSearch paramSearch = new BinarizationParamSearch(
						new CachedBinarization(originalNeuron, images, referenceImages));
				solutions[t.id] = paramSearch.searchBestLogLog();
				// synchronized (System.out) {
				// System.out.println(
				// "neuron " + t.id + ": " + solutions[t.id].getScore() /
				// originalNeuron.getMaxAgreement());
				// }
				final double relativePerf = solutions[t.id].getScore() / originalNeuron.getMaxAgreement();
				if (relativePerf < exhaustiveThreshold) {
					synchronized (System.out) {
						System.out.println("neuron " + t.id + ": marking for exhaustive (" + relativePerf + ")");
					}
					synchronized (neuronRerun) {
						neuronRerun.add(t);
						// neuronRerun.add(new Runnable() {
						//
						// @Override
						// public void run() {
						// final TernaryConfig exhaustiveSearch =
						// paramSearch.getActualBestParallel();
						// synchronized (System.out) {
						// System.out.println("neuron " + t.id + ": exhaustive
						// search changed from "
						// + relativePerf + " to "
						// + exhaustiveSearch.getScore() /
						// originalNeuron.getMaxAgreement());
						// }
						// solutions[t.id] = exhaustiveSearch;
						// }
						// });
					}

				}
				int idDone = nbDone.incrementAndGet();
				if (idDone % 100 == 0) {
					synchronized (System.out) {
						System.out.println("done " + idDone);
					}
				}
			}
		});
		System.out.println("doing exhaustive search for " + neuronRerun.size() + " neurons");
		for (RealNeuron t : neuronRerun) {
			final TanHNeuron originalNeuron = new TanHNeuron(t.weights, t.bias, false);
			final BinarizationParamSearch paramSearch = new BinarizationParamSearch(
					new CachedBinarization(originalNeuron, images, referenceImages));
			solutions[t.id] = paramSearch.getActualBestParallel();
			System.out.println("neuron " + t.id + ": exhaustive search changed to "
					+ solutions[t.id].getScore() / originalNeuron.getMaxAgreement());
		}

		BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
		for (TernaryConfig s : solutions) {
			bw.write(s.nbPosWeights + "," + s.nbNegWeights + "," + s.th + "," + s.tl + "," + s.score + "\n");
		}
		bw.close();
	}
}
