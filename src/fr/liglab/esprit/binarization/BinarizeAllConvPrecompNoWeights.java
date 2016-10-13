package fr.liglab.esprit.binarization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
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

import fr.liglab.esprit.binarization.neuron.ConvBinarizationHalfCached;
import fr.liglab.esprit.binarization.neuron.PrecompNeuron;
import fr.liglab.esprit.binarization.transformer.BinarizationParamSearch;
import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public class BinarizeAllConvPrecompNoWeights {
	private static class RealNeuronPrecomp {
		private double[] weights;
		private String activationsFile;
		private int id;
	}

	public static void main(String[] args) throws Exception {
		final Options options = new Options();
		options.addOption(Option.builder("t").longOpt("training").desc("Input training set").hasArg().argName("FILE")
				.required().build());
		options.addOption(Option.builder("r").longOpt("reference training")
				.desc("Input training set for the original neuron, to avoid propagating binarization errors").hasArg()
				.argName("FILE").required(false).build());
		options.addOption(Option.builder("w").longOpt("weights").desc("Original weights").hasArg().argName("FILE")
				.required().build());
		options.addOption(Option.builder("a").longOpt("activations").desc("Original activations").hasArg()
				.argName("FILE").required().build());
		options.addOption(Option.builder("o").longOpt("output").desc("Neuron configuration output file").hasArg()
				.argName("FILE").required().build());
		options.addOption(Option.builder("d").longOpt("consider original neuron deterministic").hasArg(false).build());
		options.addOption(
				Option.builder("ix").desc("Input horizontal size").hasArg().argName("SIZE").required().build());
		options.addOption(Option.builder("iy").desc("input vertical size").hasArg().argName("SIZE").required().build());
		options.addOption(Option.builder("ic").desc("input nb channels").hasArg().argName("SIZE").required().build());
		options.addOption(
				Option.builder("cx").desc("Convolution horizontal size").hasArg().argName("SIZE").required().build());
		options.addOption(
				Option.builder("cy").desc("Convolution vertical size").hasArg().argName("SIZE").required().build());
		options.addOption(Option.builder("cp").desc("Convolution padding").hasArg().argName("SIZE").required().build());
		options.addOption(Option.builder("imax").desc("Max val in input").hasArg().argName("VAL").required().build());
		final CommandLineParser parser = new DefaultParser();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			System.err.println(e);
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("BinarizeAll", options, true);
			System.exit(-1);
		}
		final String trainingData = cmd.getOptionValue("t");
		final String referenceTrainingData = cmd.getOptionValue("r", null);
		final String weightsData = cmd.getOptionValue("w");
		final String activationsData = cmd.getOptionValue("a");
		final String outputFile = cmd.getOptionValue("o");
		final boolean deterministic = cmd.hasOption("d");
		final int ix = Integer.parseInt(cmd.getOptionValue("ix"));
		final int iy = Integer.parseInt(cmd.getOptionValue("iy"));
		final int ic = Integer.parseInt(cmd.getOptionValue("ic"));
		final short cx = Short.parseShort(cmd.getOptionValue("cx"));
		final short cy = Short.parseShort(cmd.getOptionValue("cy"));
		final int cp = Short.parseShort(cmd.getOptionValue("cp"));
		final byte mVal = Byte.parseByte(cmd.getOptionValue("imax"));
		// double globalTw = FilesProcessing.getCentileAbsWeight(weightsData,
		// 0.80);
		final List<RealNeuronPrecomp> lNeurons = new ArrayList<>();
		final List<double[]> allWeights = FilesProcessing.getAllWeights(weightsData, Integer.MAX_VALUE);
		for (int i = 0; i < allWeights.size(); i++) {
			RealNeuronPrecomp rl = new RealNeuronPrecomp();
			rl.weights = allWeights.get(i);
			rl.activationsFile = activationsData + i;
			rl.id = i;
			lNeurons.add(rl);
		}
		final List<byte[]> images = FilesProcessing.getAllTrainingSetB(trainingData, Integer.MAX_VALUE, ix * iy * ic);
		final List<byte[]> referenceImages = (referenceTrainingData != null)
				? FilesProcessing.getAllTrainingSetB(referenceTrainingData, Integer.MAX_VALUE, ix * iy * ic) : null;
		final TernaryConfig[] solutions = new TernaryConfig[lNeurons.size()];
		final AtomicInteger nbDone = new AtomicInteger();
			lNeurons.parallelStream().forEach(new Consumer<RealNeuronPrecomp>() {

				@Override
				public void accept(final RealNeuronPrecomp t) {
					PrecompNeuron originalNeuron = null;
					try {
						originalNeuron = new PrecompNeuron(t.weights, deterministic,
								FilesProcessing.getActivationsBinary(t.activationsFile));
					} catch (IOException e) {
						e.printStackTrace();
						System.exit(-1);
					}
					final BinarizationParamSearch paramSearch = new BinarizationParamSearch(
							new ConvBinarizationHalfCached(originalNeuron, cx, cy, ix, iy, ic, cp, mVal, images,
									referenceImages));
					solutions[t.id] = paramSearch.searchExhaustiveAround(originalNeuron.getNbPosWeights(), originalNeuron.getNbNegWeights(), 0, 0);
					// synchronized (System.out) {
					// System.out.println(
					// "neuron " + t.id + ": " + solutions[t.id].getScore() /
					// originalNeuron.getMaxAgreement());
					// }
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
