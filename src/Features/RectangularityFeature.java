package Features;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.RotatedRect;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class RectangularityFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return getRectangularity(imagesFactory.GetMatrix(ImagesFactory.SegmentationV3), contoursFactory.minBoundingRect);
		
	}
	
    private Double getRectangularity (Mat segmentation, RotatedRect minBoundingRect) {
    	
    	double area = Core.countNonZero(segmentation);
        double rbBoxArea = minBoundingRect.size.width * minBoundingRect.size.height;
        double rectangularity = area / rbBoxArea;

    	return rectangularity;
    }
}
