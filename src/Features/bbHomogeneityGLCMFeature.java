package Features;

import Core.ContoursFactory;
import Core.GLCMFactory;
import Core.HistogramFactory;
import Core.IFeature;
import Core.ImagesFactory;

public class bbHomogeneityGLCMFeature implements IFeature {
	public Double getResult(ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return GLCMFactory.bbHomogeneity;
	}
}
