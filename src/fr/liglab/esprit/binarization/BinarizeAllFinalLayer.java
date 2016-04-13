package fr.liglab.esprit.binarization;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

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

	public static void main(String[] args) throws Exception {
		String trainingData = args[0];
		String weightsData = args[1];
		// String biasData = args[2];
		String groundTruthData = args[3];
		String outputFile = args[4];
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
		final TernaryConfig[] solutions = new TernaryConfig[lNeurons.size()];
		final CachedBinarization[] cachedResults = new CachedBinarization[lNeurons.size()];
		AtomicInteger nbDone = new AtomicInteger();
		lNeurons.parallelStream().forEach(new Consumer<RealNeuron>() {

			@Override
			public void accept(RealNeuron t) {
				GroundTruthNeuron originalNeuron = new GroundTruthNeuron(t.weights, groundTruth, t.id);
				cachedResults[t.id] = new CachedBinarization(originalNeuron, images);
				BinarizationParamSearch paramSearch = new BinarizationParamSearch(cachedResults[t.id]);
				solutions[t.id] = paramSearch.searchBestLogLog();
				// synchronized (System.out) {
				// System.out.println(
				// "neuron " + t.id + ": " + solutions[t.id].getScore() /
				// originalNeuron.getMaxAgreement());
				// }
				if (solutions[t.id].getScore() < 0.95) {
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
			if (Math.abs(configs[updatedNeuron].getScore() - currentPerf) < 0.001) {
				break;
			}
			BinarizationSoftMaxSearch sms = new BinarizationSoftMaxSearch(cached, configs, groundTruth, updatedNeuron);
			SoftMaxConfig config = sms.searchBestLogLog();
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
