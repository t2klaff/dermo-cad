package Features;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class ConvexityFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
        return getConvexity(imagesFactory.GetMatrix(ImagesFactory.SegmentationOutline), contoursFactory.convexHullOutline);	
	}
	
    private Double getConvexity (Mat lesionMaskOutline, Mat convexHullOutline) {
    	
        double perimeter = Core.countNonZero(lesionMaskOutline);
        double convexHullPerimeter = Core.countNonZero(convexHullOutline);
        double convexity = convexHullPerimeter / perimeter;

		if (convexity > 100.0) {
			return 0.0;
		} else {
	    	return convexity;
		}
		
//    	return convexity;
    }
}
