package fr.liglab.esprit.binarization;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class ConvolutionInputGen {

	public static void main(String[] args) throws Exception {
		final Options options = new Options();
		options.addOption(Option.builder("t").longOpt("training").desc("Input training set").hasArg().argName("FILE")
				.required().build());
		options.addOption(
				Option.builder("o").longOpt("output").desc("Output file containing input as processed by a convolution")
						.hasArg().argName("FILE").required().build());
		options.addOption(
				Option.builder("is").longOpt("input size").desc("Size of the input, for instance 28 or 32 for MNIST")
						.hasArg().argName("size").required().build());
		options.addOption(Option.builder("cs").longOpt("convolution size")
				.desc("Size of the convolution, for instance 5 for a 5x5 convolution").hasArg().argName("size")
				.required().build());
		final CommandLineParser parser = new DefaultParser();
		CommandLine cmd = null;
		try {
			cmd = parser.parse(options, args);
		} catch (ParseException e) {
			HelpFormatter formatter = new HelpFormatter();
			formatter.printHelp("ConvolutionInputGen", options, true);
			System.exit(-1);
		}
		final String trainingData = cmd.getOptionValue("t");
		final String outputFile = cmd.getOptionValue("r");
		final int imageSize = Integer.parseInt(cmd.getOptionValue("is"));
		final int convolutionSize = Integer.parseInt(cmd.getOptionValue("cs"));
		FilesProcessing.genConvolutionInput(imageSize, convolutionSize, trainingData, outputFile);
	}

}
