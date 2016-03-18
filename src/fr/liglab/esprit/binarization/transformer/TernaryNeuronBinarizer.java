package fr.liglab.esprit.binarization.transformer;

public interface TernaryNeuronBinarizer {
	public TernarySolution findBestBinarizedConfiguration();

	public void update(boolean[] input);
}