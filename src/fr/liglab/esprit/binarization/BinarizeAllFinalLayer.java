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
import fr.liglab.esprit.binarization.neuron.GroundTruthNeuron;
import fr.liglab.esprit.binarization.softmax.BinarizationSoftMaxSearch;
import fr.liglab.esprit.binarization.softmax.CachedSoftmax;
import fr.liglab.esprit.binarization.softmax.SoftMaxConfig;
import fr.liglab.esprit.binarization.transformer.BinarizationParamSearch;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public class BinarizeAllFinalLayer {
	public static class RealNeuron {
		// private double bias;
		private double[] weights;
		private int id;
	}

	private static final double DEFAULT_CONVERGENCE_THRESHOLD = 0.001;
	private static final double DEFAULT_EXHAUSTIVE_THRESHOLD = 0.95;

	public static void main(String[] args) throws Exception {
		Options options = new Options();
		options.addOption(Option.builder("t").longOpt("training").desc("Input training set").hasArg().argName("FILE")
				.required().build());
		options.addOption(Option.builder("r").longOpt("reference training")
				.desc("Input training set for the original neuron, to avoid propagating binarization errors").hasArg()
				.argName("FILE").required(false).build());
		options.addOption(Option.builder("w").longOpt("weights").desc("Original weights").hasArg().argName("FILE")
				.required().build());
		options.addOption(Option.builder("g").longOpt("groundTruth").desc("Ground truth for the training set").hasArg()
				.argName("FILE").required().build());
		options.addOption(Option.builder("o").longOpt("output").desc("Neuron configuration output file").hasArg()
				.argName("FILE").required().build());
		options.addOption(Option.builder("c").longOpt("convergence").desc(
				"Threshold to stop optimizing and assume convergence (default " + DEFAULT_CONVERGENCE_THRESHOLD + ")")
				.hasArg().argName("THRESHOLD").build());
		options.addOption(Option.builder("ei").longOpt("exhaustive initialization threshold")
				.desc("Threshold to go exhaustive in initialization (default " + DEFAULT_EXHAUSTIVE_THRESHOLD + ")")
				.hasArg().argName("THRESHOLD").build());
		options.addOption(Option.builder("ec").longOpt("exhaustive convergence")
				.desc("Flag to use exhaustive search during the convergence loop instead of loglog").build());
		CommandLineParser parser = new DefaultParser();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("BinarizeAllFinalLayer", options, true);
			System.exit(-1);
		}
		final String trainingData = cmd.getOptionValue("t");
		final String referenceTrainingData = cmd.getOptionValue("r", null);
		final String weightsData = cmd.getOptionValue("w");
		final String groundTruthData = cmd.getOptionValue("g");
		final String outputFile = cmd.getOptionValue("o");
		double convergenceThreshold = Double
				.parseDouble(cmd.getOptionValue("c", Double.toString(DEFAULT_CONVERGENCE_THRESHOLD)));
		if (convergenceThreshold < 0.) {
			throw new RuntimeException("convergence threshold must be >= 0.");
		}
		final double exhaustiveThresholdInit = Double
				.parseDouble(cmd.getOptionValue("ei", Double.toString(DEFAULT_EXHAUSTIVE_THRESHOLD)));
		if (exhaustiveThresholdInit < 0. || exhaustiveThresholdInit > 1.) {
			throw new RuntimeException("exhaustive threshold must be in [0,1]");
		}
		final boolean exhaustiveOnConvergence = cmd.hasOption("ec");
		// double globalTw = FilesProcessing.getCentileAbsWeight(weightsData,
		// 0.80);
		List<RealNeuron> lNeurons = new ArrayList<>();
		List<double[]> allWeights = FilesProcessing.getAllWeights(weightsData, Integer.MAX_VALUE);
		// List<Double> allBias = FilesProcessing.getAllBias(biasData,
		// Integer.MAX_VALUE);
		int[] groundTruth = FilesProcessing.getGroundTruth(groundTruthData, Integer.MAX_VALUE);
		for (int i = 0; i < allWeights.size(); i++) {
			RealNeuron rl = new RealNeuron();
			rl.weights = allWeights.get(i);
			// rl.bias = allBias.get(i);
			rl.id = i;
			lNeurons.add(rl);
		}
		List<byte[]> images = FilesProcessing.getAllTrainingSet(trainingData, Integer.MAX_VALUE);
		final List<byte[]> referenceImages = (referenceTrainingData != null)
				? FilesProcessing.getAllTrainingSet(referenceTrainingData, Integer.MAX_VALUE) : null;
		final TernaryConfig[] solutions = new TernaryConfig[lNeurons.size()];
		final CachedBinarization[] cachedResults = new CachedBinarization[lNeurons.size()];
		AtomicInteger nbDone = new AtomicInteger();
		lNeurons.parallelStream().forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(RealNeuron t) {
				GroundTruthNeuron originalNeuron = new GroundTruthNeuron(t.weights, groundTruth, t.id);
				cachedResults[t.id] = new CachedBinarization(originalNeuron, images, referenceImages);
				BinarizationParamSearch paramSearch = new BinarizationParamSearch(cachedResults[t.id]);
				solutions[t.id] = paramSearch.searchBestLogLog();
				// synchronized (System.out) {
				// System.out.println(
				// "neuron " + t.id + ": " + solutions[t.id].getScore() /
				// originalNeuron.getMaxAgreement());
				// }
				if (solutions[t.id].getScore() < exhaustiveThresholdInit) {
					synchronized (System.out) {
						System.out.println("neuron " + t.id + ": going	exhaustive");
					}
					TernaryConfig exhaustiveSearch = paramSearch.getActualBestParallel();
					synchronized (System.out) {
						System.out.println("neuron " + t.id + ": exhaustive search changed from "
								+ solutions[t.id].getScore() + " to " + exhaustiveSearch.getScore());
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
		SoftMaxConfig[] configs = new SoftMaxConfig[solutions.length];
		CachedSoftmax[] cached = new CachedSoftmax[solutions.length];
		for (int i = 0; i < solutions.length; i++) {
			configs[i] = new SoftMaxConfig(-solutions[i].th, solutions[i].nbPosWeights, solutions[i].nbNegWeights, -1.);
			cached[i] = new CachedSoftmax(cachedResults[i].getPosSums(), cachedResults[i].getNegSums());
		}
		double currentPerf = CachedSoftmax.getCurrentPerf(cached, configs, groundTruth);
		int updatedNeuron = 0;
		System.out.println("starting at " + currentPerf);
		while (true) {
			System.out.println("updating neuron " + updatedNeuron);
			if (Math.abs(configs[updatedNeuron].getScore() - currentPerf) <= convergenceThreshold) {
				break;
			}
			BinarizationSoftMaxSearch sms = new BinarizationSoftMaxSearch(cached, configs, groundTruth, updatedNeuron);
			SoftMaxConfig config;
			if (exhaustiveOnConvergence) {
				config = sms.getActualBestParallel();
			} else {
				config = sms.searchBestLogLog();
			}
			System.out.println("candidate version of neuron " + updatedNeuron + ":\t" + config);
			if (config.getScore() > currentPerf) {
				configs[updatedNeuron] = config;
				currentPerf = config.getScore();
			} else {
				configs[updatedNeuron].setScore(currentPerf);
			}
			updatedNeuron = (updatedNeuron + 1) % cached.length;
		}
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
		for (SoftMaxConfig s : configs) {
			bw.write(s.nbPosWeights + "," + s.nbNegWeights + "," + s.bias + "," + s.score + "\n");
		}
		bw.close();
	}
}
