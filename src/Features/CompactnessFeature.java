package Features;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class CompactnessFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return getCompactness(imagesFactory.GetMatrix(ImagesFactory.SegmentationV3), imagesFactory.GetMatrix(ImagesFactory.SegmentationOutline));
	}
	
    private Double getCompactness (Mat lesionMask, Mat lesionMaskOutline) {
    	
//    	Mat gsc = new Mat();
//	    Imgproc.cvtColor(lesionMask.clone(), gsc, Imgproc.COLOR_BGR2GRAY);
    	double area = Core.countNonZero(lesionMask);
        double perimeter = Core.countNonZero(lesionMaskOutline);
        
        double compactness = (4*Math.PI*area) / Math.pow(perimeter,2);
       
        if (compactness > 100.0) {
        	return 0.0;
        } else {
        	return compactness;
        }
        
//    	return compactness;
    }
}
