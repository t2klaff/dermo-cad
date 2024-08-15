package Features;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class SkinHistogramBMeanFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		imagesFactory.AddMatrix("bgrSkinHist", HistogramFactory.getBGRHistogram(imagesFactory.GetMatrix(imagesFactory.src), imagesFactory.GetMatrix(imagesFactory.SegmentationNot)));
		return HistogramFactory.bMeanSkin;
	}
}
