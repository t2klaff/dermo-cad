package Core;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;

public class FeatureProcessor {
	
	private List<String> _features;
	private FeatureFactory _featureFactory;
	private DiagnosisMapper _diagnosisMapper;
	
	public FeatureProcessor(List<String> features, String metadataPath) throws IOException {
		_features = features;
		_featureFactory = new FeatureFactory();
		_diagnosisMapper = new DiagnosisMapper(metadataPath);
	}
	
	public FeatureResultsAndDiagnosis getResults(String filePath, String outputFolder, boolean ds) throws IOException	{
		
		var result = new FeatureResultsAndDiagnosis();
		
		// get feature calculation results
		result.FeatureResults = GetFeatureResults(filePath, outputFolder, ds);
		// get diagnosis
		result.Diagnosis = GetDiagnosis(filePath);
		
		return result;
	}
	
	private List<Double> GetFeatureResults(String filePath, String outputFolder, boolean ds)
	{
		// get all the necessary images for feature extraction
		ImagesFactory imagesFactory = new ImagesFactory(filePath, ds);
		Mat img1 = imagesFactory.GetMatrix(ImagesFactory.SegmentationV3);
		Mat img2 = imagesFactory.GetMatrix(ImagesFactory.SegmentationBorder);
//		Mat img3 = imagesFactory.GetMatrix(ImagesFactory.SegmentationNot);
		Mat img4 = imagesFactory.GetMatrix(ImagesFactory.LesionMasked);
		Mat img5 = imagesFactory.GetMatrix(ImagesFactory.SkinMasked);

		ContoursFactory contoursFactory = new ContoursFactory(img1, img2);
		GLCMFactory GLCMFactory = new GLCMFactory(imagesFactory.GetMatrix(ImagesFactory.src), imagesFactory.GetMatrix(ImagesFactory.SegmentationV3));
		HistogramFactory HistogramFactory = new HistogramFactory(imagesFactory.GetMatrix(ImagesFactory.src), imagesFactory.GetMatrix(ImagesFactory.SegmentationV3));
		
		DecimalFormat df = new DecimalFormat("#.###");
        df.setRoundingMode(RoundingMode.CEILING);

		
		List<Double> results = new ArrayList<Double>();
		for (var f : _features)
		{
			// get the result for each feature
			Double r = GetFeatureResult(f, imagesFactory, contoursFactory, GLCMFactory, HistogramFactory);
			results.add(Double.parseDouble(df.format(r)));
		}
		
		if (outputFolder != null)
		{
			// write test images workings to output folder
			imagesFactory.WriteImages(outputFolder);
		}
		
		return results;
	}
	
	private Double GetFeatureResult(String feature, ImagesFactory imagesFactory, ContoursFactory contoursFactory, GLCMFactory GLCMFactory, HistogramFactory HistogramFactory)
	{
		return _featureFactory.getFeature(feature).getResult(imagesFactory, contoursFactory, GLCMFactory, HistogramFactory);
	}
	
	private String GetDiagnosis(String filePath) throws IOException
	{
		return _diagnosisMapper.getDiagnosis(filePath);
	}
}