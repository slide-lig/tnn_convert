package fr.liglab.esprit.binarization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import fr.liglab.esprit.binarization.neuron.ATanNeuron;
import fr.liglab.esprit.binarization.transformer.SymBinarizer;
import fr.liglab.esprit.binarization.transformer.TernarySolution;

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
		List<double[]> allWeights = FilesProcessing.getAllWeights(weightsData, Integer.MAX_VALUE);
		List<Double> allBias = FilesProcessing.getAllBias(biasData, Integer.MAX_VALUE);
		for (int i = 0; i < allWeights.size(); i++) {
			RealNeuron rl = new RealNeuron();
			rl.weights = allWeights.get(i);
			rl.bias = allBias.get(i);
			rl.id = i;
			lNeurons.add(rl);
		}
		List<boolean[]> images = FilesProcessing.getTrainingSet(trainingData, Integer.MAX_VALUE);
		final TernarySolution[] solutions = new TernarySolution[lNeurons.size()];
		AtomicInteger nbDone = new AtomicInteger();
		lNeurons.parallelStream().forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(RealNeuron t) {
				SymBinarizer transformer = new SymBinarizer(new ATanNeuron(t.weights, t.bias, true));
				for (boolean[] image : images) {
					transformer.update(image);
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
			bw.write(s.twPos + "," + s.twNeg + "," + s.th + "," + s.tl + "," + s.score + "\n");
		}
		bw.close();
	}
}
