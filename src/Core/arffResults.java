package Core;
import java.util.ArrayList;
import java.util.List;

public class arffResults {
	public List<String> Features;
	public List<FeatureResultsAndDiagnosis> Results;
	
	public arffResults() {
		Results = new ArrayList<FeatureResultsAndDiagnosis>();
	}
}
