package Features;

import org.opencv.core.RotatedRect;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class ElongationFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return getElongation(contoursFactory.minBoundingRect);
		
	}
	
	private double getElongation (RotatedRect minBoundingRect) {
    	double width = minBoundingRect.size.width;
    	double height = minBoundingRect.size.height;
    	double elongation = 0;
    	
    	if (width > height) {
    		elongation = height/width;
    	} else if (height > width) {
    		elongation = width/height;
    	} else {
    		elongation = 1;
    	}

    	return elongation;
    }
}
