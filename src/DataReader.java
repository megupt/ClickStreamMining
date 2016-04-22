
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Vector;

import weka.attributeSelection.AttributeSelection;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.ClassDiscovery.StringCompare;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
		
		
public class DataReader {

	public static String trainingData = "data\\training_subsetD.arff";
    public static String testingData = "data\\testingD.arff";
    public static HashMap<Integer, String> mostFreqValueAttr;
    static String fpPath = null;
    private static Instances GetArffData(String trainingFile)
    {
        DataSource dataSource;
        Instances data = null;
		try {
			dataSource = new DataSource(trainingFile);
	        data = dataSource.getDataSet();
	        if (data.classIndex() == -1)
	        {
	            data.setClassIndex(data.numAttributes() - 1);
	        }
		
		}
		catch(Exception e){}
        return data;
    }
    
   private static int Classify(Instance instance, TreeNode root, HashMap<String, Integer> idToName)
   {	
	   String value = instance.stringValue(instance.classIndex());
	   
	   if(root.isLeaf)
	   {
		   if (value.equals("True"))
		   {
			   if(root.classification)
				   return 1;
			   else
				   return 4;
		   }
		   else if (value.equals("False"))
		   {
			   if(root.classification)
				   return 3;
			   else
				   return 2;
		   }
	   }
	   else
	   {
		   int attrId = idToName.get(root.label);
		   String instanceAttrVal = instance.stringValue(attrId);
		   if(instance.isMissing(instance.attribute(attrId)))
		   {
			   
			   instanceAttrVal = mostFreqValueAttr.get(attrId);
		   }
		   if (root.children.containsKey(instanceAttrVal))
		   {
			   fpPath = fpPath + " " + root.label + "->" + instanceAttrVal;
			   return Classify(instance, root.children.get(instanceAttrVal), idToName);
		   }
		   else
		   {
			   if (instance.stringValue(instance.classIndex()).equals("True"))
			   {
				   return 4;
			   }
			   else if (instance.stringValue(instance.classIndex()).equals("False"))
				   return 2;
		   }
	   }
	   return -1;
   }
    private static void TestDataSet(TreeNode root)
    {
    	HashMap<String, Integer> IdToName = new HashMap<String, Integer>();
    	int TP = 0, FP = 0, TN = 0, FN = 0;
    	int positive = 0, negative = 0, failed = 0;
    	try{
    		Instances dataSet = DataReader.GetArffData(DataReader.testingData);
    		int trueCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[0];
    		System.out.println("True in test " + trueCounts);
    		for(int i =0; i< dataSet.numAttributes() - 1; i++)
    		{
    			IdToName.put(dataSet.attribute(i).name(), i);
    		}
    		HashMap<String, Integer> x = new HashMap<String, Integer>();
    		for(int i =0; i < dataSet.numInstances(); i++)
    		{
    			Instance instance = dataSet.instance(i);
    			fpPath = new String();
    			int result = Classify(instance, root, IdToName);
    			switch(result)
    			{
    				case 1: TP++;
		    				
    						break;
    				case 2: TN++;
					
    				if(!x.containsKey(fpPath))
						x.put(fpPath, 0);
					int cc = x.get(fpPath);
					x.replace(fpPath, cc + 1);
    						break;
    				case 3:	FP++;
		    				
    						break;
    				case 4: FN++;
							break;
    				case 5: failed++;
							break;
    			}
    		}
	    	
    		System.out.println("TP " + TP);
    		System.out.println("FP " + FP);
    		System.out.println("TN " + TN);
    		System.out.println("FN " + FN);
    		System.out.println("Failed " + failed);
    		
    		int hc = 0;
    		String bp = null;
    		for(String path : x.keySet())
    		{
    			if(x.get(path) > hc)
    			{
    				hc = x.get(path);
    				bp = path;
    			}
    		}
    		System.out.println("Path " + bp + " " + hc);
    		
    		
    	}catch(Exception e)
    	{
    		e.printStackTrace();
    	}
    }

    public static void main(String[] args) throws Exception 
    {        
        System.out.println("Started loading training file: " + trainingData);
        Instances dataSet = DataReader.GetArffData(trainingData);
        
        mostFreqValueAttr = new HashMap<Integer, String>();
        for(int attrIter = 0; attrIter < dataSet.numAttributes() - 1; attrIter++)
		{
        	HashMap<String, Integer> attrCounts = new HashMap<String, Integer>();
		
			for(int instanceIter = 0; instanceIter < dataSet.numInstances(); instanceIter++)
			{
				Instance currentInstance = dataSet.instance(instanceIter);
				Attribute currentAttr = currentInstance.attribute(attrIter);
				
				if(currentInstance.isMissing(currentAttr))
					continue;
				
				String valueAtSplittingAttr = currentInstance.stringValue(currentAttr);
				if(!attrCounts.containsKey(valueAtSplittingAttr))
					attrCounts.put(valueAtSplittingAttr, 0);
				
				int currentCount = attrCounts.get(valueAtSplittingAttr);
				attrCounts.replace(valueAtSplittingAttr, currentCount + 1);
			}
			
			int maxAttrCount = Integer.MIN_VALUE;
			String maxAttrVal = null;
			
			for(String key : attrCounts.keySet())
			{
				if(attrCounts.get(key) > maxAttrCount)
				{
					maxAttrVal = key;
					maxAttrCount = attrCounts.get(key);
				}
			}
			
			mostFreqValueAttr.put(attrIter, maxAttrVal);
			
		}
        
        for(int attrIter = 0; attrIter < dataSet.numAttributes() - 1; attrIter++)
		{
			HashMap<String, Instances> splitDataSet = new HashMap<String, Instances>();
			for(int instanceIter = 0; instanceIter < dataSet.numInstances(); instanceIter++)
			{
				Instance currentInstance = dataSet.instance(instanceIter);
				Attribute currentAttr = currentInstance.attribute(attrIter);
				String valueAtSplittingAttr;
				if(currentInstance.isMissing(currentAttr))
				{
					currentInstance.setValue(currentAttr, mostFreqValueAttr.get(attrIter));
				}
			}
		}
        
        System.out.println("Finished loading training file: " + DataReader.trainingData);
        DecisionTree model = new DecisionTree();
        TreeNode root = model.BuildTree(dataSet, 0);
        System.out.println("root node is " + root.label + " " + root.labelid);
        TestDataSet(root);
    }
}
