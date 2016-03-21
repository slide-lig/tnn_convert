package fr.liglab.esprit.binarization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import fr.liglab.esprit.binarization.neuron.TanHNeuron;
import fr.liglab.esprit.binarization.transformer.SparsitySymBinarizer;
import fr.liglab.esprit.binarization.transformer.TernaryNeuronBinarizer;
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
		double globalTw = FilesProcessing.getCentileAbsWeight(weightsData, 0.80);
		List<RealNeuron> lNeurons = new ArrayList<>();
		List<double[]> allWeights = FilesProcessing.getFilteredWeights(weightsData, Integer.MAX_VALUE);
		List<Double> allBias = FilesProcessing.getAllBias(biasData, Integer.MAX_VALUE);
		for (int i = 0; i < allWeights.size(); i++) {
			RealNeuron rl = new RealNeuron();
			rl.weights = allWeights.get(i);
			rl.bias = allBias.get(i);
			rl.id = i;
			lNeurons.add(rl);
		}
		List<boolean[]> images = FilesProcessing.getFilteredTrainingSet(trainingData, Integer.MAX_VALUE);
		final TernarySolution[][] solutions = new TernarySolution[lNeurons.size()][ScoreFunctions.values().length];
		AtomicInteger nbDone = new AtomicInteger();
		lNeurons.parallelStream().forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(RealNeuron t) {
				TernaryNeuronBinarizer transformer = new SparsitySymBinarizer(new TanHNeuron(t.weights, t.bias, false),
						globalTw);
				for (boolean[] image : images) {
					transformer.update(image);
				}
				solutions[t.id] = transformer.findBestBinarizedConfiguration(ScoreFunctions.values());
				int idDone = nbDone.incrementAndGet();
				if (idDone % 10 == 0) {
					System.out.println("done " + idDone);
				}
			}
		});
		for (int i = 0; i < ScoreFunctions.values().length; i++) {
			BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile + "-" + ScoreFunctions.values()[i]));
			for (TernarySolution[] sList : solutions) {
				TernarySolution s = sList[i];
				bw.write(s.twPos + "," + s.twNeg + "," + s.th + "," + s.tl + "," + s.score + "\n");
			}
			bw.close();
		}
	}
}
