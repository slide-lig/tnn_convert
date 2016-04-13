package fr.liglab.esprit.binarization.transformer;

import fr.liglab.esprit.binarization.ScoreFunctions;

public interface TernaryNeuronBinarizer {
	public TernarySolution[] findBestBinarizedConfiguration(ScoreFunctions[] scoreFuns);

	public void update(byte[] input);
}