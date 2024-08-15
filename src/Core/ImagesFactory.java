package Core;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import ImageProcessors.BlackBordersImageProcessor;
import ImageProcessors.HairMaskImageProcessor;
import ImageProcessors.InpaintedImageProcessor;
import ImageProcessors.SegmentationImageProcessorV1;
import ImageProcessors.SegmentationImageProcessorV2;
import ImageProcessors.SegmentationImageProcessorV3;

public class ImagesFactory {
	private String _fileName;
	private Mat _sourceImage;
	private HashMap<String, Mat> _matrices;
	private HashMap<String, Mat> _additionalMatrices;
	private Size bigSize = new Size(1280, 1280);
	
	public static final String src = "src";
	public static final String srcDs = "srcDs";
	
	public static final String BorderRemoved = "borderRemoved";
	public static final String BorderRemovedDs = "borderRemovedDs";
	
	public static final String HairMask = "hairMask";
	public static final String HairMaskDs = "hairMaskDs";

	public static final String Inpainted = "inpainted";
	public static final String InpaintedDs = "inpaintedDs";

	public static final String SegmentationV1 = "segmentationV1";
	public static final String SegmentationV2 = "segmentationV2";
	public static final String SegmentationV3 = "segmentationV3";
	
	public static final String SegmentationDs = "segmentationDs";
	public static final String SegmentationBorder = "segmentationBorder";
	public static final String SegmentationOutline = "segmentationOutline";
	public static final String SegmentationOutlineBorder = "segmentationOutlineBorder";
	
	public static final String SegmentationNot = "segmentationNot";
	public static final String LesionMasked = "lesionMasked";
	public static final String SkinMasked = "skinMasked";

	
	public ImagesFactory(String filePath, boolean downscaled)
	{
		Path path = Paths.get(filePath);
		_fileName = path.getFileName().toString();
		if (downscaled == true) {
			Mat og = Imgcodecs.imread(filePath);
			Mat ds = obtainDownscaled(og);
			_sourceImage = ds.clone();
			
		} else {
			_sourceImage = Imgcodecs.imread(filePath);
		}
		_matrices = new HashMap<String, Mat>();
		_additionalMatrices = new HashMap<String, Mat>();
	}
	
	public Mat GetMatrix(String name)
	{
		if (!_matrices.containsKey(name))
		{
			switch (name)
			{
				case src:  _matrices.put(name, _sourceImage); break;
				
				case BorderRemoved:  _matrices.put(name, new BlackBordersImageProcessor().getMatrix(_sourceImage)); break;
			
				case HairMask:  _matrices.put(name, new HairMaskImageProcessor().getMatrix(GetMatrix(BorderRemoved))); break;
					
				case Inpainted:  _matrices.put(name, new InpaintedImageProcessor().getMatrix(GetMatrix(BorderRemoved), GetMatrix(HairMask))); break;
					
				case SegmentationV1:  _matrices.put(name, new SegmentationImageProcessorV1().getMatrix(GetMatrix(Inpainted))); break;
				case SegmentationV2:  _matrices.put(name, new SegmentationImageProcessorV2().getMatrix(GetMatrix(Inpainted))); break;
				case SegmentationV3:  _matrices.put(name, new SegmentationImageProcessorV3().getMatrix(GetMatrix(Inpainted))); break;
				
				case SegmentationBorder:  _matrices.put(name, obtainBlackBorders(GetMatrix(SegmentationV3))); break;
				case SegmentationOutline:  _matrices.put(name, obtainOutline(GetMatrix(SegmentationV3))); break;
				case SegmentationOutlineBorder:  _matrices.put(name, obtainOutline(GetMatrix(SegmentationBorder))); break;
				
				case SegmentationNot:  _matrices.put(name, obtainBitwiseNot(GetMatrix(SegmentationV3))); break;
				case LesionMasked: _matrices.put(name, obtainMasked(GetMatrix(Inpainted), GetMatrix(SegmentationV3))); break;
				case SkinMasked: _matrices.put(name, obtainMasked(GetMatrix(Inpainted), GetMatrix(SegmentationNot).clone())); break;
			}
		}
		return _matrices.get(name);
	}

	public void AddMatrix(String name, Mat matrix)
	{
		_additionalMatrices.put(name, matrix);
	}

	public void WriteImages(String folderName)
	{
		String imageFolder = folderName +  "\\" + _fileName;
		CreateFolder(imageFolder);
//		HashMap<String, Mat> mats = new HashMap<>();
//		mats.put("segmentation", GetMatrix(SegmentationV3));
//		WriteImages(folderName, _fileName, mats);
		WriteImages(imageFolder, _fileName, _matrices);
		WriteImages(imageFolder, _fileName, _additionalMatrices);
	}
	
	private void CreateFolder(String imageFolder)
	{
	    File directory = new File(imageFolder);
	    if (! directory.exists()){
	        directory.mkdir();
	    }
	}
	
	private void WriteImages(String folderName, String fileName, HashMap<String, Mat> matrices)
	{
		// create folder
		for (String key : matrices.keySet()) {
			var imageName = folderName + "\\" + getWithoutExtension(fileName) + "_" + key + ".png";
//			System.out.println("writing to file: " + fileName);
			Imgcodecs.imwrite(imageName, matrices.get(key));
		}
	}
	
    private Mat obtainBlackBorders (Mat src) {
    	double missingHeight = bigSize.height - src.rows();
    	double missingWidth = bigSize.width - src.cols();
        int addTop = (int) Math.floor(missingHeight/2);
        int addBottom = (int) Math.ceil(missingHeight/2);
        int addLeft = (int) Math.floor(missingWidth/2);
        int addRight = (int) Math.ceil(missingWidth/2);

    	Mat bordered = new Mat();
        Core.copyMakeBorder(src, bordered, addTop, addBottom, addLeft, addRight, Core.BORDER_CONSTANT, new Scalar(0,0,0));
//        System.out.println(bordered.size());
    	return bordered;
    }
    
    private Mat obtainOutline (Mat src) {
    	Mat outline = new Mat();
        Imgproc.Canny(src, outline, 0, 255);
        return outline;
    }
    
    private Mat obtainMasked (Mat src, Mat mask) {
    	Mat masked = new Mat();
    	Imgproc.cvtColor(mask, mask, Imgproc.COLOR_GRAY2BGR);
    	Core.bitwise_and(src, mask, masked);
        return masked;
    }
    
    private Mat obtainBitwiseNot (Mat mask) {
    	Mat res = mask.clone();
//    	if (mask.type() == CvType.CV_8UC3) {
//    		Imgproc.cvtColor(mask.clone(), res, Imgproc.COLOR_BGR2GRAY);
//    	}
    	Mat bitwiseNot = new Mat(res.size(), res.type());
    	Core.bitwise_not(res, bitwiseNot);
		Imgproc.cvtColor(bitwiseNot, bitwiseNot, Imgproc.COLOR_BGR2GRAY);
    	return bitwiseNot;
    }
    
    private Mat obtainDownscaled (Mat mat) {
    	Mat resized = new Mat();
    	Size scaleSize = new Size(mat.width()/2, mat.height()/2);
    	Imgproc.resize(mat, resized, scaleSize, Imgproc.INTER_AREA);
    	return resized;
    }
    
    private String getWithoutExtension(String fileFullPath){
        return fileFullPath.substring(0, fileFullPath.lastIndexOf('.'));
    }

}
