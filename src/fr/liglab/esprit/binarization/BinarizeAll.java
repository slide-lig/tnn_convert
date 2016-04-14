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
		Options options = new Options();
		options.addOption(Option.builder("t").longOpt("training").desc("Input training set").hasArg().argName("FILE")
				.required().build());
		options.addOption(Option.builder("w").longOpt("weights").desc("Original weights").hasArg().argName("FILE")
				.required().build());
		options.addOption(
				Option.builder("b").longOpt("bias").desc("Original bias").hasArg().argName("FILE").required().build());
		options.addOption(Option.builder("o").longOpt("output").desc("Neuron configuration output file").hasArg()
				.argName("FILE").required().build());
		options.addOption(Option.builder("e").longOpt("exhaustive")
				.desc("Threshold to go exhaustive (default " + DEFAULT_EXHAUSTIVE_THRESHOLD + ")").hasArg()
				.argName("THRESHOLD").build());
		CommandLineParser parser = new DefaultParser();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("BinarizeAll", options, true);
			System.exit(-1);
		}
		String trainingData = cmd.getOptionValue("t");
		String weightsData = cmd.getOptionValue("w");
		String biasData = cmd.getOptionValue("b");
		String outputFile = cmd.getOptionValue("o");
		double exhaustiveThreshold = Double
				.parseDouble(cmd.getOptionValue("e", Double.toString(DEFAULT_EXHAUSTIVE_THRESHOLD)));
		if (exhaustiveThreshold < 0. || exhaustiveThreshold > 1.) {
			throw new RuntimeException("exhaustive threshold must be in [0,1]");
		}
		// double globalTw = FilesProcessing.getCentileAbsWeight(weightsData,
		// 0.80);
		List<RealNeuron> lNeurons = new ArrayList<>();
		List<double[]> allWeights = FilesProcessing.getAllWeights(weightsData, Integer.MAX_VALUE);
		List<Double> allBias = FilesProcessing.getAllBias(biasData, Integer.MAX_VALUE);
		for (int i = 0; i < allWeights.size(); i++) {
			RealNeuron rl = new RealNeuron();
			rl.weights = allWeights.get(i);
			rl.bias = allBias.get(i);
			rl.id = i;
			lNeurons.add(rl);
		}
		List<byte[]> images = FilesProcessing.getAllTrainingSet(trainingData, Integer.MAX_VALUE);
		final TernaryConfig[] solutions = new TernaryConfig[lNeurons.size()];
		AtomicInteger nbDone = new AtomicInteger();
		lNeurons.parallelStream().forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(RealNeuron t) {
				TanHNeuron originalNeuron = new TanHNeuron(t.weights, t.bias, false);
				BinarizationParamSearch paramSearch = new BinarizationParamSearch(
						new CachedBinarization(originalNeuron, images));
				solutions[t.id] = paramSearch.searchBestLogLog();
				// synchronized (System.out) {
				// System.out.println(
				// "neuron " + t.id + ": " + solutions[t.id].getScore() /
				// originalNeuron.getMaxAgreement());
				// }
				if (solutions[t.id].getScore() / originalNeuron.getMaxAgreement() < exhaustiveThreshold) {
					// synchronized (System.out) {
					// System.out.println("neuron " + t.id + ": going
					// exhaustive");
					// }
					TernaryConfig exhaustiveSearch = paramSearch.getActualBestParallel();
					synchronized (System.out) {
						System.out.println("neuron " + t.id + ": exhaustive search changed from "
								+ solutions[t.id].getScore() / originalNeuron.getMaxAgreement() + " to "
								+ exhaustiveSearch.getScore() / originalNeuron.getMaxAgreement());
					}
					solutions[t.id] = exhaustiveSearch;
				}
				int idDone = nbDone.incrementAndGet();
				if (idDone % 100 == 0) {
					synchronized (System.out) {
						System.out.println("done " + idDone);
					}
				}
			}
		});
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
		for (TernaryConfig s : solutions) {
			bw.write(s.nbPosWeights + "," + s.nbNegWeights + "," + s.th + "," + s.tl + "," + s.score + "\n");
		}
		bw.close();
	}
}
