package Core;
import java.util.ArrayList;
import java.util.List;

import Features.CircularityFeature;
import Features.CompactnessFeature;
import Features.ConvexityFeature;
import Features.ElongationFeature;
import Features.FractalDimensionFeature;
import Features.LesionHistogramAMeanFeature;
import Features.LesionHistogramAMedianFeature;
import Features.LesionHistogramAStdFeature;
import Features.LesionHistogramBMeanFeature;
import Features.LesionHistogramBMedianFeature;
import Features.LesionHistogramBStdFeature;
import Features.LesionHistogramBbMeanFeature;
import Features.LesionHistogramBbMedianFeature;
import Features.LesionHistogramBbStdFeature;
import Features.LesionHistogramGMeanFeature;
import Features.LesionHistogramGMedianFeature;
import Features.LesionHistogramGStdFeature;
import Features.LesionHistogramLMeanFeature;
import Features.LesionHistogramLMedianFeature;
import Features.LesionHistogramLStdFeature;
import Features.LesionHistogramRMeanFeature;
import Features.LesionHistogramRMedianFeature;
import Features.LesionHistogramRStdFeature;
import Features.RadialVarianceFeature;
import Features.RectangularityFeature;
import Features.SkinHistogramAMeanFeature;
import Features.SkinHistogramAMedianFeature;
import Features.SkinHistogramAStdFeature;
import Features.SkinHistogramBMeanFeature;
import Features.SkinHistogramBMedianFeature;
import Features.SkinHistogramBStdFeature;
import Features.SkinHistogramBbMeanFeature;
import Features.SkinHistogramBbMedianFeature;
import Features.SkinHistogramBbStdFeature;
import Features.SkinHistogramGMeanFeature;
import Features.SkinHistogramGMedianFeature;
import Features.SkinHistogramGStdFeature;
import Features.SkinHistogramLMeanFeature;
import Features.SkinHistogramLMedianFeature;
import Features.SkinHistogramLStdFeature;
import Features.SkinHistogramRMeanFeature;
import Features.SkinHistogramRMedianFeature;
import Features.SkinHistogramRStdFeature;
import Features.SolidityFeature;
import Features.SymmetryFeature;
import Features.bConstrastGLCMFeature;
import Features.bCorrelationGLCMFeature;
import Features.bEnergyGLCMFeature;
import Features.bEntropyGLCMFeature;
import Features.bHomogeneityGLCMFeature;
import Features.bMeanGLCMFeature;
import Features.bVarianceGLCMFeature;
import Features.gConstrastGLCMFeature;
import Features.gCorrelationGLCMFeature;
import Features.gEnergyGLCMFeature;
import Features.gEntropyGLCMFeature;
import Features.gHomogeneityGLCMFeature;
import Features.gMeanGLCMFeature;
import Features.gVarianceGLCMFeature;
import Features.rConstrastGLCMFeature;
import Features.rCorrelationGLCMFeature;
import Features.rEnergyGLCMFeature;
import Features.rEntropyGLCMFeature;
import Features.rHomogeneityGLCMFeature;
import Features.rMeanGLCMFeature;
import Features.rVarianceGLCMFeature;

public class FeatureFactory {
	
	public List<String> getFeatures() {
		List<String> features = new ArrayList<>();
		features.add("asymmetry");
		features.add("circularity");
		features.add("compactness");
		features.add("convexity");
		features.add("elongation");
		features.add("rectangularity");
		features.add("solidity");
		features.add("fractalDimension");
		features.add("radialVariance");
		features.add("lesionBMean");
		features.add("lesionGMean");
		features.add("lesionRMean");
		features.add("lesionBMedian");
		features.add("lesionGMedian");
		features.add("lesionRMedian");
		features.add("lesionBStd");
		features.add("lesionGStd");
		features.add("lesionRStd");
		features.add("skinBMean");
		features.add("skinGMean");
		features.add("skinRMean");
		features.add("skinBMedian");
		features.add("skinGMedian");
		features.add("skinRMedian");
		features.add("skinBStd");
		features.add("skinGStd");
		features.add("skinRStd");
		features.add("lesionLMean");
		features.add("lesionAMean");
		features.add("lesionBbMean");
		features.add("lesionLMedian");
		features.add("lesionAMedian");
		features.add("lesionBbMedian");
		features.add("lesionLStd");
		features.add("lesionAStd");
		features.add("lesionBbStd");
		features.add("skinLMean");
		features.add("skinAMean");
		features.add("skinBbMean");
		features.add("skinLMedian");
		features.add("skinAMedian");
		features.add("skinBbMedian");
		features.add("skinLStd");
		features.add("skinAStd");
		features.add("skinBbStd");
		
		features.add("bContrast");
		features.add("bCorrelation");
		features.add("bEnergy");
		features.add("bEntropy");
		features.add("bHomogeneity");
		features.add("bMean");
		features.add("bVariance");
		features.add("gContrast");
		features.add("gCorrelation");
		features.add("gEnergy");
		features.add("gEntropy");
		features.add("gHomogeneity");
		features.add("gMean");
		features.add("gVariance");
		features.add("rContrast");
		features.add("rCorrelation");
		features.add("rEnergy");
		features.add("rEntropy");
		features.add("rHomogeneity");
		features.add("rMean");
		features.add("rVariance");
		features.add("lContrast");
		features.add("lCorrelation");
		features.add("lEnergy");
		features.add("lEntropy");
		features.add("lHomogeneity");
		features.add("lMean");
		features.add("lVariance");
		features.add("aContrast");
		features.add("aCorrelation");
		features.add("aEnergy");
		features.add("aEntropy");
		features.add("aHomogeneity");
		features.add("aMean");
		features.add("aVariance");
		features.add("bbContrast");
		features.add("bbCorrelation");
		features.add("bbEnergy");
		features.add("bbEntropy");
		features.add("bbHomogeneity");
		features.add("bbMean");
		features.add("bbVariance");

		return features;
	}
	
	public IFeature getFeature(String feature) {
		switch(feature)
		{
			case "asymmetry" : return new SymmetryFeature();
			case "circularity" : return new CircularityFeature();
			case "compactness" : return new CompactnessFeature();
			case "convexity" : return new ConvexityFeature();
			case "elongation" : return new ElongationFeature();
			case "rectangularity" : return new RectangularityFeature();
			case "solidity" : return new SolidityFeature();
			case "fractalDimension" : return new FractalDimensionFeature();
			case "radialVariance" : return new RadialVarianceFeature();
			case "lesionBMean" : return new LesionHistogramBMeanFeature();
			case "lesionGMean" : return new LesionHistogramGMeanFeature();
			case "lesionRMean" : return new LesionHistogramRMeanFeature();
			case "lesionBMedian" : return new LesionHistogramBMedianFeature();
			case "lesionGMedian" : return new LesionHistogramGMedianFeature();
			case "lesionRMedian" : return new LesionHistogramRMedianFeature();
			case "lesionBStd" : return new LesionHistogramBStdFeature();
			case "lesionGStd" : return new LesionHistogramGStdFeature();
			case "lesionRStd" : return new LesionHistogramRStdFeature();
			case "skinBMean" : return new SkinHistogramBMeanFeature();
			case "skinGMean" : return new SkinHistogramGMeanFeature();
			case "skinRMean" : return new SkinHistogramRMeanFeature();
			case "skinBMedian" : return new SkinHistogramBMedianFeature();
			case "skinGMedian" : return new SkinHistogramGMedianFeature();
			case "skinRMedian" : return new SkinHistogramRMedianFeature();
			case "skinBStd" : return new SkinHistogramBStdFeature();
			case "skinGStd" : return new SkinHistogramGStdFeature();
			case "skinRStd" : return new SkinHistogramRStdFeature();
			
			case "lesionLMean" : return new LesionHistogramLMeanFeature();
			case "lesionAMean" : return new LesionHistogramAMeanFeature();
			case "lesionBbMean" : return new LesionHistogramBbMeanFeature();
			case "lesionLMedian" : return new LesionHistogramLMedianFeature();
			case "lesionAMedian" : return new LesionHistogramAMedianFeature();
			case "lesionBbMedian" : return new LesionHistogramBbMedianFeature();
			case "lesionLStd" : return new LesionHistogramLStdFeature();
			case "lesionAStd" : return new LesionHistogramAStdFeature();
			case "lesionBbStd" : return new LesionHistogramBbStdFeature();
			case "skinLMean" : return new SkinHistogramLMeanFeature();
			case "skinAMean" : return new SkinHistogramAMeanFeature();
			case "skinBbMean" : return new SkinHistogramBbMeanFeature();
			case "skinLMedian" : return new SkinHistogramLMedianFeature();
			case "skinAMedian" : return new SkinHistogramAMedianFeature();
			case "skinBbMedian" : return new SkinHistogramBbMedianFeature();
			case "skinLStd" : return new SkinHistogramLStdFeature();
			case "skinAStd" : return new SkinHistogramAStdFeature();
			case "skinBbStd" : return new SkinHistogramBbStdFeature();
			
			case "bContrast" : return new bConstrastGLCMFeature();
			case "bCorrelation" : return new bCorrelationGLCMFeature();
			case "bEnergy" : return new bEnergyGLCMFeature();
			case "bEntropy" : return new bEntropyGLCMFeature();
			case "bHomogeneity" : return new bHomogeneityGLCMFeature();
			case "bMean" : return new bMeanGLCMFeature();
			case "bVariance" : return new bVarianceGLCMFeature();
			case "gContrast" : return new gConstrastGLCMFeature();
			case "gCorrelation" : return new gCorrelationGLCMFeature();
			case "gEnergy" : return new gEnergyGLCMFeature();
			case "gEntropy" : return new gEntropyGLCMFeature();
			case "gHomogeneity" : return new gHomogeneityGLCMFeature();
			case "gMean" : return new gMeanGLCMFeature();
			case "gVariance" : return new gVarianceGLCMFeature();
			case "rContrast" : return new rConstrastGLCMFeature();
			case "rCorrelation" : return new rCorrelationGLCMFeature();
			case "rEnergy" : return new rEnergyGLCMFeature();
			case "rEntropy" : return new rEntropyGLCMFeature();
			case "rHomogeneity" : return new rHomogeneityGLCMFeature();
			case "rMean" : return new rMeanGLCMFeature();
			case "rVariance" : return new rVarianceGLCMFeature();
			
			case "lContrast" : return new bConstrastGLCMFeature();
			case "lCorrelation" : return new bCorrelationGLCMFeature();
			case "lEnergy" : return new bEnergyGLCMFeature();
			case "lEntropy" : return new bEntropyGLCMFeature();
			case "lHomogeneity" : return new bHomogeneityGLCMFeature();
			case "lMean" : return new bMeanGLCMFeature();
			case "lVariance" : return new bVarianceGLCMFeature();
			case "aContrast" : return new gConstrastGLCMFeature();
			case "aCorrelation" : return new gCorrelationGLCMFeature();
			case "aEnergy" : return new gEnergyGLCMFeature();
			case "aEntropy" : return new gEntropyGLCMFeature();
			case "aHomogeneity" : return new gHomogeneityGLCMFeature();
			case "aMean" : return new gMeanGLCMFeature();
			case "aVariance" : return new gVarianceGLCMFeature();
			case "bbContrast" : return new rConstrastGLCMFeature();
			case "bbCorrelation" : return new rCorrelationGLCMFeature();
			case "bbEnergy" : return new rEnergyGLCMFeature();
			case "bbEntropy" : return new rEntropyGLCMFeature();
			case "bbHomogeneity" : return new rHomogeneityGLCMFeature();
			case "bbMean" : return new rMeanGLCMFeature();
			case "bbVariance" : return new rVarianceGLCMFeature();
			
		}
		return null;
	}
}
