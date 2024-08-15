package Core;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;

public class arffWriter {

	private FileWriter _writer;
	
	public void WriteResults(arffResults results, String outputFileName) throws IOException {
    	_writer = new FileWriter(outputFileName, false);
    	
    	WriteHeader(results.Features);
    	for (var item : results.Results) {
    		_writerow(item);
    	}
    	
        _writer.flush();
        _writer.close();	}
	
	private void WriteHeader(List<String> features) throws IOException {
    	_writer.write("@RELATION skin_lesions\n");
   	
    	for (var f : features) {
    		_writer.write("@ATTRIBUTE " + f + "					NUMERIC \n");
    	}

    	_writer.write("@ATTRIBUTE class			   	   {MALIGNANT,BENIGN} \n");
    	_writer.write("@data											\n");
	}
	
	private void _writerow(FeatureResultsAndDiagnosis item) throws IOException {
        String commaSeparatedStr = item.FeatureResults.stream().map(String::valueOf).collect(Collectors.joining(","));
        _writer.write(commaSeparatedStr + "," + item.Diagnosis + " \n");		
	}
}