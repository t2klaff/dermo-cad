package Core;
import java.io.File;
import java.io.IOException;
import java.util.List;

import org.opencv.core.Core;

public class Coordinator {
	public void run(String trainingFolder, String testFolder, String outputFolder, boolean ds) throws Exception {
	    // Load the native OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		String trainingArffFilePath = trainingFolder + "\\training.arff";
		String trainingMetadataPath = ".\\data\\ISIC_2019_Training_GroundTruth.csv";
//		String trainingMetadataPath = trainingFolder + "\\HAM10000_metadata";
		String testArffFilePath = testFolder + "\\test.arff";
		File f = new File(trainingArffFilePath);
		
		// if no training ARFF file exists...
		if(!f.isFile())
 		{
			// process any images and write the ARFF
			System.out.println("Processing training folder...");
			processFolder(trainingFolder, trainingArffFilePath, trainingMetadataPath, null, ds);
			System.out.println("arff written: [" + trainingFolder + "\\training.arff]");
			System.out.println("");
		}
		
		// if there are any images in the test folder...
		if (testFolder != null)
		{
			// process the images and write the ARFF
			System.out.println("Processing test folder...");
			processFolder(testFolder, testArffFilePath, null, outputFolder, ds);
			System.out.println("arff written: [" + testFolder + "\\test.arff]");
			System.out.println("");
		}
		
		new MachineLearningController().getMachineLearning(trainingArffFilePath, outputFolder);

//		new SkinTypeMapper().mapBySkinType();
//		new SkinTypeSorter().sortBySkinType("hamSkinTypeMap.txt","Training_HAM10000", "HAM10000_by_skintype");
//		new SkinTypeSorter().sortBySkinType("bcnSkinTypeMap.txt","Training_BCN20000", "BCN20000_by_skintype");
//		new SkinTypeSorter().sortBySkinType("mskSkinTypeMap.txt","Training_MSK", "MSK_by_skintype");
		
//		new SegmentationComparison().getComparisonIoU(trainingFolder);
	}
	
	private void processFolder(String folderName, String outputFileName, String metadataPath, String outputFolder, boolean ds) throws IOException
	{
		arffResults results = new arffResults();
		
		// get the features to be computed
		FeatureFactory featureListProvider = new FeatureFactory();
		results.Features = featureListProvider.getFeatures(); 
		
		FeatureProcessor featureProcessor = new FeatureProcessor(results.Features, metadataPath);
		
		long startTotalTime = System.nanoTime();
		int imgCount = 0;
		// get the files in the specified folder
		List<String> FileList = new FileListProvider().getFiles(folderName);
		for (var f : FileList)
		{
			imgCount++;
    		long startTime = System.nanoTime();
    		
//			System.out.println(f);
    		// get the feature calculation results and any available ground truth
			results.Results.add(featureProcessor.getResults(f, outputFolder, ds));
			
            long endTime = System.nanoTime();
            long duration = (endTime - startTime)/1000000;
    		System.out.println("I: " + imgCount + "/" + FileList.size() + "  	" + (f.substring(f.lastIndexOf("\\") + 1)) + " 	T: " + duration + "ms");
		}
        long endTotalTime = System.nanoTime();
        long durationTotal = (endTotalTime - startTotalTime)/1000000000;
		System.out.println("total time: " + durationTotal + "s");

		// write the results to ARFF file
		new arffWriter().WriteResults(results, outputFileName);
	}
}