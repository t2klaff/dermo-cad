package Core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.math.RoundingMode;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class SegmentationComparison {
	
	private String _curSegmentationPath = "E:\\Final Year Project Dataset\\HAM10000 Segmentations\\Curated";
	private String _segmentationV1Path = "E:\\Final Year Project Dataset\\HAM10000 Segmentations\\V1";
	private String _segmentationV2Path = "E:\\Final Year Project Dataset\\HAM10000 Segmentations\\V2";
	private String _segmentationV3Path = "E:\\Final Year Project Dataset\\HAM10000 Segmentations\\V3";
	
	public void getComparisonIoU (String trainingFolder) throws IOException {
		
		List<String> FileList = new FileListProvider().getFiles(trainingFolder);
		DecimalFormat df = new DecimalFormat("#.###");
        df.setRoundingMode(RoundingMode.CEILING);

		Map<String, List<Double>> iouMap = new HashMap<>();
		
		Double iouV1Sum = 0.0;
		Double iouV2Sum = 0.0;
		Double iouV3Sum = 0.0;
		
		int count = 1;
		for (var f : FileList)
		{
			String fileName = getFileName(f);
			ImagesFactory imagesFactory = new ImagesFactory(f, false);
			Mat segmentationV1 = imagesFactory.GetMatrix(ImagesFactory.SegmentationV1);
			Mat segmentationV2 = imagesFactory.GetMatrix(ImagesFactory.SegmentationV2);
			Mat segmentationV3 = imagesFactory.GetMatrix(ImagesFactory.SegmentationV3);
			Mat segmentationCur = Imgcodecs.imread(_curSegmentationPath + "\\" + fileName + "_segmentation.png", Imgcodecs.IMREAD_GRAYSCALE);
			Double iouV1 = getIoU(segmentationV1, segmentationCur);
			Double iouV2 = getIoU(segmentationV2, segmentationCur);
			Double iouV3 = getIoU(segmentationV3, segmentationCur);
			List<Double> iouList = Arrays.asList(iouV1, iouV2, iouV3);
			iouMap.put(fileName, iouList);
			iouV1Sum += iouV1;
			iouV2Sum += iouV2;
			iouV3Sum += iouV3;
			
			System.out.println(count + "/" + FileList.size() + " - " + fileName + " - " + "V1: "+Double.parseDouble(df.format(iouV1Sum / count)) + ", V2: "+Double.parseDouble(df.format(iouV2Sum / count)) + ", V3: "+Double.parseDouble(df.format(iouV3Sum / count)));
			
			Imgcodecs.imwrite(_segmentationV1Path + "\\" + fileName + "_segmentation.png", segmentationV1);
			Imgcodecs.imwrite(_segmentationV2Path + "\\" + fileName + "_segmentation.png", segmentationV2);
			Imgcodecs.imwrite(_segmentationV3Path + "\\" + fileName + "_segmentation.png", segmentationV3);
			count++;
		}
		
		Double iouV1Mean = iouV1Sum / iouMap.size();
		Double iouV2Mean = iouV2Sum / iouMap.size();
		Double iouV3Mean = iouV3Sum / iouMap.size();
		
		Double iouStdV1 = 0.0;
		Double iouStdV2 = 0.0;
		Double iouStdV3 = 0.0;
		
		for (var r : iouMap.entrySet()) {
			List<Double> iouList = r.getValue();
			iouStdV1 += Math.pow((iouList.get(0) - iouV1Mean), 2);
			iouStdV2 += Math.pow((iouList.get(1) - iouV2Mean), 2);
			iouStdV3 += Math.pow((iouList.get(2) - iouV3Mean), 2);
		}
		
		
		try (BufferedWriter bf = new BufferedWriter(new FileWriter("C:\\Users\\toml4\\Documents\\Uni\\YEAR 3\\CM3203 - FINAL YEAR PROJECT\\Project\\Segmentation Comparison\\iouResults.txt"))) {
            bf.write("img,v1,v2,v3");
            bf.newLine();
			for (Map.Entry<String, List<Double>> entry : iouMap.entrySet()) {
				List<Double> results = entry.getValue();
                bf.write(entry.getKey() + "," + results.get(0) + "," + results.get(1) + "," + results.get(2));
                bf.newLine();
            }
            bf.write("-------------------------");
            bf.newLine();
            bf.write("V1 mean: " + iouV1Mean);
            bf.newLine();
            bf.write("V2 mean: " + iouV2Mean);
            bf.newLine();
            bf.write("V3 mean: " + iouV3Mean);
            bf.newLine();
            bf.write("V1 std: " + iouStdV1);
            bf.newLine();
            bf.write("V2 std: " + iouStdV2);
            bf.newLine();
            bf.write("V3 std: " + iouStdV3);
            bf.flush();
        }
	}
	
	private Double getIoU (Mat src1, Mat src2) {
		
//		Imgproc.cvtColor(src1, src1, Imgproc.COLOR_BGR2GRAY);
//		Imgproc.cvtColor(src2, src2, Imgproc.COLOR_BGR2GRAY);
		
		Mat intersection = new Mat();
        Core.bitwise_and(src1, src2, intersection);
        Mat union = new Mat();
        Core.bitwise_or(src1, src2, union);

        int count_white_own= Core.countNonZero(src1);
        int count_white_cur= Core.countNonZero(src2);
        double count_white_intersection= Core.countNonZero(intersection);
        double count_white_union= Core.countNonZero(union);
        double intersection_over_union = (count_white_intersection / count_white_union);
        
        return intersection_over_union;
	}
	
	private String getFileName(String filePath) {
		Path path = Paths.get(filePath);
		String fileName = path.getFileName().toString();
		return fileName.substring(0, fileName.lastIndexOf('.'));
	}
	
}
