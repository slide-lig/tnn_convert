package fr.liglab.esprit.binarization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.transformer.ASymBinarizer;
import fr.liglab.esprit.binarization.transformer.TernaryNeuronBinarizer;
import fr.liglab.esprit.binarization.transformer.TernarySolution;

public class BinarizeAllParallelInput {
	public static class RealNeuron {
		private double bias;
		private double[] weights;
		// private int id;
	}

	public static void main(String[] args) throws Exception {
		String trainingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		String outputFile = args[3];
		List<RealNeuron> lNeurons = new ArrayList<>();
		List<double[]> allWeights = FilesProcessing.getFilteredWeights(weightsData, 1);
		List<Double> allBias = FilesProcessing.getAllBias(biasData, Integer.MAX_VALUE);
		for (int i = 0; i < allWeights.size(); i++) {
			RealNeuron rl = new RealNeuron();
			rl.weights = allWeights.get(i);
			rl.bias = allBias.get(i);
			// rl.id = i;
			lNeurons.add(rl);
		}
		List<boolean[]> images = FilesProcessing.getFilteredTrainingSet(trainingData, 1000);
		long startTime = System.currentTimeMillis();
		// final TernarySolution[] solutions = new
		// TernarySolution[lNeurons.size()];
		AtomicInteger nbDone = new AtomicInteger();
		BufferedWriter[] bw = new BufferedWriter[ScoreFunctions.values().length];
		for (int i = 0; i < ScoreFunctions.values().length; i++) {
			bw[i] = new BufferedWriter(new FileWriter(outputFile + "-" + ScoreFunctions.values()[i]));
		}
		lNeurons.forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(RealNeuron t) {
				TernaryNeuronBinarizer transformer = new ASymBinarizer(new TanHNeuron(t.weights, t.bias, false), true);
				images.parallelStream().forEach(image -> transformer.update(image));
				TernarySolution[] solutions = transformer.findBestBinarizedConfiguration(ScoreFunctions.values());
				try {
					for (int i = 0; i < solutions.length; i++) {
						TernarySolution s = solutions[i];
						bw[i].write(s.twPos + "," + s.twNeg + "," + s.th + "," + s.tl + "," + s.score + "\n");
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
				int done = nbDone.incrementAndGet();
				if (done % 1 == 0) {
					System.out.println("done " + done);
					try {
						for (BufferedWriter b : bw) {
							b.flush();
						}
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
		});

		// for (TernarySolution s : solutions) {
		// bw.write(s.twPos + "," + s.twNeg + "," + s.th + "," + s.tl + "," +
		// s.score + "\n");
		// }
		for (BufferedWriter b : bw) {
			b.close();
		}
		System.out.println((System.currentTimeMillis() - startTime) / 1000 + " seconds");
	}
}
