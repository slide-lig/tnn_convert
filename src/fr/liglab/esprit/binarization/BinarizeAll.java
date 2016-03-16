package fr.liglab.esprit.binarization;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import fr.liglab.esprit.binarization.BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer.TernarySolution;

public class BinarizeAll {
	public static class RealNeuron {
		private double bias;
		private double[] weights;
		private int id;
	}

	public static void main(String[] args) throws Exception {
		String trainingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		String outputFile = args[3];
		List<RealNeuron> lNeurons = new ArrayList<>();
		// load neuron weights
		BufferedReader br = new BufferedReader(new FileReader(weightsData));
		String line;
		int lineNumber = 0;
		while ((line = br.readLine()) != null) {
			String[] input = line.split(",");
			double[] weights = new double[input.length];
			for (int i = 0; i < input.length; i++) {
				weights[i] = Double.parseDouble(input[i]);
			}
			RealNeuron rl = new RealNeuron();
			rl.weights = weights;
			lNeurons.add(rl);
			rl.id = lineNumber;
			lineNumber++;
		}
		br.close();
		// load bias
		br = new BufferedReader(new FileReader(biasData));
		lineNumber = 0;
		Iterator<RealNeuron> iter = lNeurons.iterator();
		while ((line = br.readLine()) != null) {
			double bias = Double.valueOf(line);
			iter.next().bias = bias;
			lineNumber++;
		}
		br.close();
		List<boolean[]> images = new ArrayList<>();
		// read input
		br = new BufferedReader(new FileReader(trainingData));
		int nbSamp = 0;
		while ((line = br.readLine()) != null) {
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
				images.add(input);
			}
			nbSamp++;
		}
		br.close();
		final TernarySolution[] solutions = new TernarySolution[lNeurons.size()];
		AtomicInteger nbDone = new AtomicInteger();
		lNeurons.parallelStream().forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(RealNeuron t) {
				double[] probBuffer = new double[3];
				BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer transformer = new BinaryInputTernaryWeightsAtanStochasticNeuronBinarizer(
						t.weights, t.bias);
				for (boolean[] image : images) {
					transformer.getOutputProbs(image, probBuffer);
					transformer.updateBinaryAccuracy(image, probBuffer);
				}
				solutions[t.id] = transformer.findBestBinarizedConfiguration();
				int idDone = nbDone.incrementAndGet();
				if (idDone % 10 == 0) {
					System.out.println("done " + idDone);
				}
			}
		});
		BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile));
		for (TernarySolution s : solutions) {
			bw.write(s.tw + "," + s.th + "," + s.tl + "," + s.score + "\n");
		}
		bw.close();
	}
}
