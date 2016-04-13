package fr.liglab.esprit.binarization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

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

	public static void main(String[] args) throws Exception {
		String trainingData = args[0];
		String weightsData = args[1];
		String biasData = args[2];
		String outputFile = args[3];
		// double globalTw = FilesProcessing.getCentileAbsWeight(weightsData,
		// 0.80);
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
				if (solutions[t.id].getScore() / originalNeuron.getMaxAgreement() < 0.95) {
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
