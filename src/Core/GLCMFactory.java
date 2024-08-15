package Core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class GLCMFactory {
	public double contrast;
	public double correlation;
	public double energy;
	public double entropy;
	public double homogeneity;
	public double maxProb;
	public double mean;
	public double variance;
	
	public double bContrast;
	public double bCorrelation;
	public double bEnergy;
	public double bEntropy;
	public double bHomogeneity;
	public double bMean;
	public double bVariance;
	public double gContrast;
	public double gCorrelation;
	public double gEnergy;
	public double gEntropy;
	public double gHomogeneity;
	public double gMean;
	public double gVariance;
	public double rContrast;
	public double rCorrelation;
	public double rEnergy;
	public double rEntropy;
	public double rHomogeneity;
	public double rMean;
	public double rVariance;
	
	public double lContrast;
	public double lCorrelation;
	public double lEnergy;
	public double lEntropy;
	public double lHomogeneity;
	public double lMean;
	public double lVariance;
	public double aContrast;
	public double aCorrelation;
	public double aEnergy;
	public double aEntropy;
	public double aHomogeneity;
	public double aMean;
	public double aVariance;
	public double bbContrast;
	public double bbCorrelation;
	public double bbEnergy;
	public double bbEntropy;
	public double bbHomogeneity;
	public double bbMean;
	public double bbVariance;


	public GLCMFactory(Mat src, Mat segmentation)
	{
//        Mat gsc = new Mat();
//        Imgproc.cvtColor(src, gsc, Imgproc.COLOR_BGR2GRAY);
        Map<String, Double> averageBGRGLCM = getAverageGLCM(src, segmentation); 
        bContrast = averageBGRGLCM.get("ch1Contrast");
    	bCorrelation = averageBGRGLCM.get("ch1Correlation");
    	bEnergy = averageBGRGLCM.get("ch1Energy");
    	bEntropy = averageBGRGLCM.get("ch1Entropy");
    	bHomogeneity = averageBGRGLCM.get("ch1Homogeneity");
    	bMean = averageBGRGLCM.get("ch1Mean");
    	bVariance = averageBGRGLCM.get("ch1Variance");
        gContrast = averageBGRGLCM.get("ch2Contrast");
    	gCorrelation = averageBGRGLCM.get("ch2Correlation");
    	gEnergy = averageBGRGLCM.get("ch2Energy");
    	gEntropy = averageBGRGLCM.get("ch2Entropy");
    	gHomogeneity = averageBGRGLCM.get("ch2Homogeneity");
    	gMean = averageBGRGLCM.get("ch2Mean");
    	gVariance = averageBGRGLCM.get("ch2Variance");
        rContrast = averageBGRGLCM.get("ch3Contrast");
    	rCorrelation = averageBGRGLCM.get("ch3Correlation");
    	rEnergy = averageBGRGLCM.get("ch3Energy");
    	rEntropy = averageBGRGLCM.get("ch3Entropy");
    	rHomogeneity = averageBGRGLCM.get("ch3Homogeneity");
    	rMean = averageBGRGLCM.get("ch3Mean");
    	rVariance = averageBGRGLCM.get("ch3Variance");
        
        Mat lab = new Mat();
        Imgproc.cvtColor(src.clone(), lab, Imgproc.COLOR_BGR2Lab);
        Map<String, Double> averageLabGLCM = getAverageGLCM(lab, segmentation);
        lContrast = averageLabGLCM.get("ch1Contrast");
    	lCorrelation = averageLabGLCM.get("ch1Correlation");
    	lEnergy = averageLabGLCM.get("ch1Energy");
    	lEntropy = averageLabGLCM.get("ch1Entropy");
    	lHomogeneity = averageLabGLCM.get("ch1Homogeneity");
    	lMean = averageLabGLCM.get("ch1Mean");
    	lVariance = averageLabGLCM.get("ch1Variance");
        aContrast = averageLabGLCM.get("ch2Contrast");
    	aCorrelation = averageLabGLCM.get("ch2Correlation");
    	aEnergy = averageLabGLCM.get("ch2Energy");
    	aEntropy = averageLabGLCM.get("ch2Entropy");
    	aHomogeneity = averageLabGLCM.get("ch2Homogeneity");
    	aMean = averageLabGLCM.get("ch2Mean");
    	aVariance = averageLabGLCM.get("ch2Variance");
        bbContrast = averageLabGLCM.get("ch3Contrast");
    	bbCorrelation = averageLabGLCM.get("ch3Correlation");
    	bbEnergy = averageLabGLCM.get("ch3Energy");
    	bbEntropy = averageLabGLCM.get("ch3Entropy");
    	bbHomogeneity = averageLabGLCM.get("ch3Homogeneity");
    	bbMean = averageLabGLCM.get("ch3Mean");
    	bbVariance = averageLabGLCM.get("ch3Variance");
        
	}
	
	private Map<String, Double> getAverageGLCM (Mat src, Mat mask) {
		Mat ch1 = new Mat();
		Core.extractChannel(src, ch1, 0);
		Mat ch2 = new Mat();
		Core.extractChannel(src, ch2, 1);
		Mat ch3 = new Mat();
		Core.extractChannel(src, ch3, 2);
        Mat ch1Masked = new Mat();
        Core.bitwise_and(ch1, mask, ch1Masked);
        Mat ch2Masked = new Mat();
        Core.bitwise_and(ch2, mask, ch2Masked);
        Mat ch3Masked = new Mat();
        Core.bitwise_and(ch3, mask, ch3Masked);

		// get glcm for each channel and for each angle
		List<Mat> glcmCh1List = new ArrayList<>();
		glcmCh1List.add(getNormalizedGLCM(ch1Masked, 0));
		glcmCh1List.add(getNormalizedGLCM(ch1Masked, 45));
		glcmCh1List.add(getNormalizedGLCM(ch1Masked, 90));
		glcmCh1List.add(getNormalizedGLCM(ch1Masked, 135));
		
		List<Mat> glcmCh2List = new ArrayList<>();
		glcmCh2List.add(getNormalizedGLCM(ch2Masked, 0));
		glcmCh2List.add(getNormalizedGLCM(ch2Masked, 45));
		glcmCh2List.add(getNormalizedGLCM(ch2Masked, 90));
		glcmCh2List.add(getNormalizedGLCM(ch2Masked, 135));
		
		List<Mat> glcmCh3List = new ArrayList<>();
		glcmCh3List.add(getNormalizedGLCM(ch3Masked, 0));
		glcmCh3List.add(getNormalizedGLCM(ch3Masked, 45));
		glcmCh3List.add(getNormalizedGLCM(ch3Masked, 90));
		glcmCh3List.add(getNormalizedGLCM(ch3Masked, 135));
		
		double ch1ContrastSum = 0;
		double ch1CorrelationSum = 0;
		double ch1EnergySum = 0;
		double ch1EntropySum = 0;
		double ch1HomogeneitySum = 0;
		double ch1MeanSum = 0;
		double ch1VarianceSum = 0;
		double ch2ContrastSum = 0;
		double ch2CorrelationSum = 0;
		double ch2EnergySum = 0;
		double ch2EntropySum = 0;
		double ch2HomogeneitySum = 0;
		double ch2MeanSum = 0;
		double ch2VarianceSum = 0;
		double ch3ContrastSum = 0;
		double ch3CorrelationSum = 0;
		double ch3EnergySum = 0;
		double ch3EntropySum = 0;
		double ch3HomogeneitySum = 0;
		double ch3MeanSum = 0;
		double ch3VarianceSum = 0;
		
		for (Mat glcm : glcmCh1List) {
			Map<String, Double> glcmFeatures = getGLCMFeaturesMap(glcm);
			ch1ContrastSum = ch1ContrastSum + glcmFeatures.get("contrast");
			ch1CorrelationSum = ch1CorrelationSum + glcmFeatures.get("correlation");
			ch1EnergySum = ch1EnergySum + glcmFeatures.get("energy");
			ch1EntropySum = ch1EntropySum + glcmFeatures.get("entropy");
			ch1HomogeneitySum = ch1HomogeneitySum + glcmFeatures.get("homogeneity");
			ch1MeanSum = ch1MeanSum + glcmFeatures.get("mean");
			ch1VarianceSum = ch1VarianceSum + glcmFeatures.get("variance");
		}
		
		for (Mat glcm : glcmCh2List) {
			Map<String, Double> glcmFeatures = getGLCMFeaturesMap(glcm);
			ch2ContrastSum = ch2ContrastSum + glcmFeatures.get("contrast");
			ch2CorrelationSum = ch2CorrelationSum + glcmFeatures.get("correlation");
			ch2EnergySum = ch2EnergySum + glcmFeatures.get("energy");
			ch2EntropySum = ch2EntropySum + glcmFeatures.get("entropy");
			ch2HomogeneitySum = ch2HomogeneitySum + glcmFeatures.get("homogeneity");
			ch2MeanSum = ch2MeanSum + glcmFeatures.get("mean");
			ch2VarianceSum = ch2VarianceSum + glcmFeatures.get("variance");
		}
		
		for (Mat glcm : glcmCh3List) {
			Map<String, Double> glcmFeatures = getGLCMFeaturesMap(glcm);
			ch3ContrastSum = ch3ContrastSum + glcmFeatures.get("contrast");
			ch3CorrelationSum = ch3CorrelationSum + glcmFeatures.get("correlation");
			ch3EnergySum = ch3EnergySum + glcmFeatures.get("energy");
			ch3EntropySum = ch3EntropySum + glcmFeatures.get("entropy");
			ch3HomogeneitySum = ch3HomogeneitySum + glcmFeatures.get("homogeneity");
			ch3MeanSum = ch3MeanSum + glcmFeatures.get("mean");
			ch3VarianceSum = ch3VarianceSum + glcmFeatures.get("variance");
		}
		
		Map<String, Double> averageGLCM = new HashMap<String, Double>();
		
		averageGLCM.put("ch1Contrast", ch1ContrastSum/glcmCh1List.size());
		averageGLCM.put("ch1Correlation", ch1CorrelationSum/glcmCh1List.size());
		averageGLCM.put("ch1Energy", ch1EnergySum/glcmCh1List.size());
		averageGLCM.put("ch1Entropy", ch1EntropySum/glcmCh1List.size());
		averageGLCM.put("ch1Homogeneity", ch1HomogeneitySum/glcmCh1List.size());
		averageGLCM.put("ch1Mean", ch1MeanSum/glcmCh1List.size());
		averageGLCM.put("ch1Variance", ch1VarianceSum/glcmCh1List.size());
		averageGLCM.put("ch2Contrast", ch2ContrastSum/glcmCh2List.size());
		averageGLCM.put("ch2Correlation", ch2CorrelationSum/glcmCh2List.size());
		averageGLCM.put("ch2Energy", ch2EnergySum/glcmCh2List.size());
		averageGLCM.put("ch2Entropy", ch2EntropySum/glcmCh2List.size());
		averageGLCM.put("ch2Homogeneity", ch2HomogeneitySum/glcmCh2List.size());
		averageGLCM.put("ch2Mean", ch2MeanSum/glcmCh2List.size());
		averageGLCM.put("ch2Variance", ch2VarianceSum/glcmCh2List.size());
		averageGLCM.put("ch3Contrast", ch3ContrastSum/glcmCh3List.size());
		averageGLCM.put("ch3Correlation", ch3CorrelationSum/glcmCh3List.size());
		averageGLCM.put("ch3Energy", ch3EnergySum/glcmCh3List.size());
		averageGLCM.put("ch3Entropy", ch3EntropySum/glcmCh3List.size());
		averageGLCM.put("ch3Homogeneity", ch3HomogeneitySum/glcmCh3List.size());
		averageGLCM.put("ch3Mean", ch3MeanSum/glcmCh3List.size());
		averageGLCM.put("ch3Variance", ch3VarianceSum/glcmCh3List.size());
		
		return averageGLCM;
	}
	
	private Mat getNormalizedGLCM (Mat img, int angle) {
    	Mat gl = Mat.zeros(256, 256, CvType.CV_64F);
    	Mat glt = gl.clone();
    	
    	int startX = 0;
    	int startY = 0;
    	int endX = img.cols();
    	int endY = img.rows();
    	int jChangeX = 0;
    	int jChangeY = 0;
    	
    	// change pixel co-occurrance search offset for each angle
    	switch (angle) {
	    	case 0:
	    		endX = endX - 1;
	    		jChangeX = jChangeX + 1;  
	    		break;
	    	case 45:
	    		startY = startY + 1;
	    		endX = endX - 1;
	    		jChangeY = jChangeY - 1;
	    		jChangeX = jChangeX + 1;
	    		break;
	    	case 90:
	    		startY = startY + 1;
	    		jChangeY = jChangeY - 1;
	    		break;
	    	case 135:
	    		startY = startY + 1;
	    		startX = startX + 1;
	    		jChangeY = jChangeY - 1;
	    		jChangeX = jChangeX - 1;
	    		break;
    	} 
    	
    	// do the searching for co-occurrances, building the glcm
    	for (int y = startY; y < endY; y++) {
    	    for (int x = startX; x < endX; x++) {
    	        int i = (int) img.get(y, x)[0];
    	        if (i != 0) {
        	        int j = (int) img.get(y + jChangeY, x + jChangeX)[0];
        	        if (j != 0) {
        	        	double[] count = gl.get(i, j);
            	        count[0]++;
            	        gl.put(i, j, count);
        	        }
    	        }
    	    }
    	}      
    	
    	// normalise the glcm
    	Core.transpose(gl, glt);
    	Core.add(gl, glt, gl);
    	Scalar gl0Sum = Core.sumElems(gl);
    	Core.divide(gl, gl0Sum, gl); 	
    	
		return gl;
	}
	
	private Map<String, Double> getGLCMFeaturesMap (Mat glcm) {
		double asm = 0;
    	double contrast = 0;
    	double correlation = 0;
    	double energy = 0;
    	double entropy = 0;
    	double homogeneity = 0;
    	double maxProb = 0;
    	double mean = 0;
    	double iMean = 0;
    	double jMean = 0;
    	double variance = 0;
    	double iVariance = 0;
    	double jVariance = 0;
    	
    	// search the glcm a few times for all the features 
    	for(int i=0;i<256;i++) {
    	    for(int j=0;j<256;j++) {
    	    	asm = asm + glcm.get(i,j)[0] * glcm.get(i,j)[0];
    	    	contrast = contrast + (i-j) * (i-j) * glcm.get(i,j)[0];
    	    	homogeneity = homogeneity + glcm.get(i,j)[0] / (1 + ((i-j)*(i-j)));
    	        if(glcm.get(i,j)[0] != 0) {
    	        	entropy = entropy - glcm.get(i,j)[0] * Math.log10(glcm.get(i,j)[0]);
    	        	if(glcm.get(i,j)[0] > maxProb) {
    	        		maxProb = glcm.get(i,j)[0];
    	        	}
    	        }
    	        iMean = iMean + (i*glcm.get(i,j)[0]);
    	        jMean = jMean + (j*glcm.get(i,j)[0]);
	        }
    	}
    	for(int i=0;i<256;i++) {
    	    for(int j=0;j<256;j++) {
    	    	iVariance = iVariance + (i-iMean) * (i-iMean) * glcm.get(i,j)[0];
    	    	jVariance = jVariance + (j-jMean) * (j-jMean) * glcm.get(i,j)[0];
	        }
    	}
    	for(int i=0;i<256;i++) {
    	    for(int j=0;j<256;j++) {
    	    	correlation = correlation + (((i-iMean) * (j-jMean))/Math.sqrt(iVariance*jVariance)) * glcm.get(i,j)[0];
	        }
    	}
    	energy = Math.sqrt(asm);
    	mean = (iMean+jMean)/2;
    	variance = (iVariance+jVariance)/2;
    	
    	Map<String, Double> GLCMFeatures = new HashMap<String, Double>();
    	List<Double> glcmFeatures = new ArrayList<>();
    	glcmFeatures.add(contrast);
    	glcmFeatures.add(correlation);
    	glcmFeatures.add(energy);
    	glcmFeatures.add(entropy);
    	glcmFeatures.add(homogeneity);
    	glcmFeatures.add(mean);
    	glcmFeatures.add(variance);
    	GLCMFeatures.put("contrast", contrast);
    	GLCMFeatures.put("correlation", correlation);
    	GLCMFeatures.put("energy", energy);
    	GLCMFeatures.put("entropy", entropy);
    	GLCMFeatures.put("homogeneity", homogeneity);
//    	GLCMFeatures.put("maxProb", maxProb);
    	GLCMFeatures.put("mean", mean);
    	GLCMFeatures.put("variance", variance);
    	
    	
		return GLCMFeatures;
	}
   
}
