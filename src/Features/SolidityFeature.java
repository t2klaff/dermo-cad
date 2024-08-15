package Features;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class SolidityFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return getSolidity(imagesFactory.GetMatrix(ImagesFactory.SegmentationV3), contoursFactory.convexHullFilled);
		
	}
	
    private Double getSolidity (Mat segmentation, Mat convexHullFilled) {
    	
    	double area = Core.countNonZero(segmentation);
    	double convexHullArea = Core.countNonZero(convexHullFilled);
        double solidity = area / convexHullArea;

    	return solidity;
    }
}
