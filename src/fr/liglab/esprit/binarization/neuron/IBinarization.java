package fr.liglab.esprit.binarization.neuron;

import fr.liglab.esprit.binarization.transformer.TernaryConfig;

public interface IBinarization {

	TernaryConfig getBestConfig(int nbPosWeights, int nbNegWeights);

	int getNbPosPossibilities();

	int getNbNegPossibilities();

	int getInputSize();

}