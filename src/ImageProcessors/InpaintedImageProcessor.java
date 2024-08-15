package ImageProcessors;

import org.opencv.core.Mat;
import org.opencv.photo.Photo;

public class InpaintedImageProcessor {
	public Mat getMatrix(Mat src, Mat hairMask)
	{
		
//		Imgproc.cvtColor(hairMask, hairMask, Imgproc.COLOR_BGR2GRAY);
    	Mat inpainted = new Mat();
        Photo.inpaint(src, hairMask, inpainted, 3, Photo.INPAINT_TELEA);
//	    HighGui.imshow("inpainted", inpainted);
//	    HighGui.waitKey();
    	return inpainted;	
    	}
}
