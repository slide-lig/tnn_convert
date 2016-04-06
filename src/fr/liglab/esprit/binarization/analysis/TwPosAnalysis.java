package fr.liglab.esprit.binarization.analysis;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import fr.liglab.esprit.binarization.FilesProcessing;

public class TwPosAnalysis {
	public static void main(String[] args) throws Exception {
		List<double[]> weights = FilesProcessing.getFilteredWeights(
				"/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/sw1.txt", Integer.MAX_VALUE);
		Iterator<double[]> weightsIter = weights.iterator();
		List<Double> posSparsityList = new ArrayList<>();
		List<Double> negSparsityList = new ArrayList<>();
		List<Double> globalSparsityList = new ArrayList<>();
		List<Double> twPosList = new ArrayList<>();
		List<Double> twNegList = new ArrayList<>();
		List<List<Double>> allLists = new ArrayList<>();
		allLists.add(posSparsityList);
		allLists.add(negSparsityList);
		allLists.add(globalSparsityList);
		allLists.add(twPosList);
		allLists.add(twNegList);
		BufferedReader br = new BufferedReader(new FileReader(
				"/Users/vleroy/workspace/esprit/mnist_binary/StochasticWeights/binary_agreement_asym.txt"));
		String line;
		while ((line = br.readLine()) != null) {
			String[] split = line.split(",");
			double twPos = Double.parseDouble(split[0]);
			double twNeg = Double.parseDouble(split[1]);
			twPosList.add(twPos);
			twNegList.add(twNeg);
			double[] neuronWeights = weightsIter.next();
			Arrays.sort(neuronWeights);
			int twPosIndex = Arrays.binarySearch(neuronWeights, twPos);
			int twNegIndex = Arrays.binarySearch(neuronWeights, twNeg);
			int nbNegWeights = Arrays.binarySearch(neuronWeights, 0.);
			if (nbNegWeights < 0) {
				// res=(-(insertion point) - 1)
				nbNegWeights = -nbNegWeights - 1;
			}
			int nbPosWeights = neuronWeights.length - nbNegWeights;
			double posSparsity = (neuronWeights.length - (twPosIndex + 1)) / ((double) nbPosWeights);
			double negSparsity = twNegIndex / ((double) nbNegWeights);
			double globalSparsity = 1. - (twPosIndex - twNegIndex + 1) / ((double) neuronWeights.length);
			posSparsityList.add(posSparsity);
			negSparsityList.add(negSparsity);
			globalSparsityList.add(globalSparsity);
		}
		br.close();
		for (List<Double> lD : allLists) {
			Collections.sort(lD);
			System.out.println(
					lD.get(0) + "\t" + lD.get((int) (lD.size() * 0.25)) + "\t" + lD.get((int) (lD.size() * 0.5)) + "\t"
							+ lD.get((int) (lD.size() * 0.75)) + "\t" + lD.get(lD.size() - 1));
		}
	}
}
