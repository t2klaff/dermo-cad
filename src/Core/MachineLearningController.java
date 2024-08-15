package Core;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;

public class MachineLearningController {
	
//	private String _trainingDataPath;
//	private String _testDataPath;
	
	public void getMachineLearning(String trainingDataPath, String output) throws Exception {
		
		System.out.println("performing classification...");
		
		// create training data from source
		DataSource sourceTraining = new DataSource(trainingDataPath);
		
		// normalise attributes
		Instances dataTraining = sourceTraining.getDataSet();
		Normalize norm = new Normalize();
		norm.setInputFormat(dataTraining);
		Instances dataTrainingNorm = Filter.useFilter(dataTraining, norm);
		if (dataTrainingNorm.classIndex() == -1)
			dataTrainingNorm.setClassIndex(dataTrainingNorm.numAttributes() - 1);
		
		// get smote class balanced
		Instances dataTrainingSMOTE = getSmote(dataTrainingNorm);
		
		runClassifiers(dataTrainingSMOTE, output);
		System.out.println("results written: [" + output + "\\10xValResults.csv]");
	}
	
	private void runClassifiers(Instances data, String output) throws Exception {
		
		// get frequency of each class
		List<Integer> classFreq = getClassFreq(data);
		
		int benign = classFreq.get(1);
		int malignant = classFreq.get(0);
		int total = classFreq.get(2);
		
		// get results for each classifier
		List<String> kNN1Results = getkNNResults(data, String.valueOf(1), 10);
		List<String> kNN5Results = getkNNResults(data, String.valueOf(5), 10);
		List<String> kNN11Results = getkNNResults(data, String.valueOf(11), 10);
		
		List<String> bayesNetResults = getBayesNetResults(data, 10);
		
		List<String> randomForestResults = getRandomForestResults(data, 10);

		// write results to file
		try (BufferedWriter bf = new BufferedWriter(new FileWriter(output + "\\10xValResults.csv"))) {
            
			bf.write("SMOTE");
            bf.newLine();
			bf.write("benign:" + "," + benign);
            bf.newLine();
			bf.write("malignant:" + "," + malignant);
            bf.newLine();
			bf.write("total:" + "," + total);
            bf.newLine();
            bf.newLine();
			
			for (String s : kNN1Results) {
				bf.write(s);
                bf.newLine();
			}
            bf.newLine();
            
			for (String s : kNN5Results) {
				bf.write(s);
                bf.newLine();
			}
            bf.newLine();
            
			for (String s : kNN11Results) {
				bf.write(s);
                bf.newLine();
			}
            bf.newLine();
            
			for (String s : bayesNetResults) {
				bf.write(s);
                bf.newLine();
			}
            bf.newLine();
            
			for (String s : randomForestResults) {
				bf.write(s);
                bf.newLine();
			}
            bf.newLine();
            
            bf.flush();
		}
	}
	
	private List<Integer> getClassFreq (Instances data) {
		
		int nbClass = data.numClasses();
		int[] instancePerClass = new int[nbClass];
		int[] labels = new int[nbClass];
		int[] classIndex = new int[nbClass];
		data.sort(data.classAttribute());
		for (int i = 0; i < nbClass; i++) {
			instancePerClass[i] = data.attributeStats(data.classIndex()).nominalCounts[i];
			labels[i] = i;
			if (i > 0)
				classIndex[i] = classIndex[i-1] + instancePerClass[i-1];
		}
		
		return Arrays.asList(instancePerClass[0], instancePerClass[1], (instancePerClass[0]+instancePerClass[1]));
	}
	
	
	private List<String> getkNNResults (Instances data, String k, int x) throws Exception {
		
		String classifierName = "kNN ("+k+")";
		
		IBk kNN = new IBk();
		String optionskNN = ( " -K "+k+" " );
		String[] optionskNNArray = optionskNN.split( " " );
		kNN.setOptions(optionskNNArray);
		System.out.println("	kNN("+k+")...");
		kNN.buildClassifier(data);
		
		System.out.println("		10 fold cross validating...");
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(kNN, data, x, new Random(1));
		
		return getResultsArray(eval, classifierName);
	}
	
	private List<String> getBayesNetResults (Instances data, int x) throws Exception {
		
		String classifierName = "BayesNet";
		
		BayesNet bayes = new BayesNet();
		String optionsBayesNet = ( " -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5 " );
		String[] optionsBayesNetArray = optionsBayesNet.split( " " );
		bayes.setOptions(optionsBayesNetArray);
		System.out.println("	BayesNet...");
		bayes.buildClassifier(data);
		
		System.out.println("		10 fold cross validating...");
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(bayes, data, x, new Random(1));
		
		return getResultsArray(eval, classifierName);
	}
	
	private List<String> getRandomForestResults (Instances data, int x) throws Exception {
		
		String classifierName = "Random Forest";
		
		RandomForest randomForest = new RandomForest();
		String optionsRandomForest = ( " -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 " );
		String[] optionsRandomForestArray = optionsRandomForest.split( " " );
		randomForest.setOptions(optionsRandomForestArray);
		System.out.println("	RandomForest...");
		randomForest.buildClassifier(data);
		
		System.out.println("		10 fold cross validating...");
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(randomForest, data, x, new Random(1));
		
		return getResultsArray(eval, classifierName);
	}
	
	private List<String> getResultsArray (Evaluation eval, String classifierName) {
		
		DecimalFormat df3dp = new DecimalFormat("#.###");
		df3dp.setRoundingMode(RoundingMode.CEILING);
		DecimalFormat df0dp = new DecimalFormat("#");
		df0dp.setRoundingMode(RoundingMode.HALF_DOWN);
		
		List<String> results = new ArrayList<>();
		results.add(classifierName + "," + "accuracy:" + "," + df0dp.format((eval.pctCorrect())) + "%");
//		results.add("xVal-10" + "," + "kappa:" + "," + df3dp.format(eval.kappa()));
//		results.add(" " + "," + "MAE:" + "," + df3dp.format(eval.meanAbsoluteError()));
//		results.add(" " + "," + "RMSE:" + "," + df3dp.format(eval.rootMeanSquaredError()));
		results.add(" " + "," + "precision:" + "," + df3dp.format(eval.weightedPrecision()));
		results.add(" " + "," + "recall:" + "," + df3dp.format(eval.weightedRecall()));
		results.add(" " + "," + "F1:" + "," + df3dp.format(eval.weightedFMeasure()));
		results.add(" " + "," + "ROC:" + "," + df3dp.format(eval.weightedAreaUnderROC()));
		results.add(" " + "," + "MCC:" + "," + df3dp.format(eval.weightedMatthewsCorrelation()));
		results.add("confusion matrix:" + "," + "mal" + "," + "ben");
		results.add("mal" + "," + df0dp.format(eval.confusionMatrix()[0][0]) + "," + df0dp.format(eval.confusionMatrix()[0][1]));
		results.add("ben" + "," + df0dp.format(eval.confusionMatrix()[1][0]) + "," + df0dp.format(eval.confusionMatrix()[1][1]));
		
		return results;
	}
	
	private Instances getSmote(Instances data) throws Exception {
		
		DecimalFormat df3dp = new DecimalFormat("#.##");
		df3dp.setRoundingMode(RoundingMode.CEILING);
		
		List<Integer> classFreq = getClassFreq(data);
		
		int remainder = classFreq.get(1) - classFreq.get(0);
		
		Double perc = 0.01 + (100.0 * (Double.valueOf(remainder) / Double.valueOf(classFreq.get(0))));
//		System.out.println(Double.parseDouble(df3dp.format(perc)));
		
		SMOTE smote = new SMOTE();
		String optionsSMOTE = ( " -S 1 -P " + perc + " -K 5 -C 0 " );
		String[] optionsSMOTEArray = optionsSMOTE.split( " " );
		smote.setOptions(optionsSMOTEArray);
		smote.setInputFormat(data);
//		System.out.println("SMOTE class balancing...");
		Instances dataTrainingSMOTE = Filter.useFilter(data, smote);
		
		return dataTrainingSMOTE;
	}
	
}
