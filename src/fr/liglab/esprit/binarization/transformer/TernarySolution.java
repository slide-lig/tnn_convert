package fr.liglab.esprit.binarization.transformer;

public class TernarySolution {
	public final int th;
	public final int tl;
	public final double twPos;
	public final double twNeg;
	public final int twPosIndex;
	public final int twNegIndex;
	public final TernaryConfusionMatrix confusionMat;
	public final double score;

	public TernarySolution(int th, int tl, double twPos, double twNeg, int twPosIndex, int twNegIndex,
			TernaryConfusionMatrix confusionMat, double score) {
		super();
		this.th = th;
		this.tl = tl;
		this.twPos = twPos;
		this.twNeg = twNeg;
		this.twPosIndex = twPosIndex;
		this.twNegIndex = twNegIndex;
		this.confusionMat = confusionMat;
		this.score = score;
	}

	@Override
	public String toString() {
		return "TernarySolution [th=" + th + ", tl=" + tl + ", twPos=" + twPos + ", twNeg=" + twNeg
				+ ", twPosIndex=" + twPosIndex + ", twNegIndex=" + twNegIndex + ", confusionMat=" + confusionMat
				+ ", score=" + score + "]";
	}

}