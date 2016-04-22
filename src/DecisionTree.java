import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class DecisionTree{

	public TreeNode BuildTree(Instances dataSet, int level)
	{
		System.out.println("in level " + level);
		//All values in the instances belong to the same class
		String lastInstanceClass = null;
		boolean sameClass = true;
		for(int instanceIterator = 0; instanceIterator < dataSet.numInstances(); instanceIterator++)
		{
			if(lastInstanceClass == null)
			{
				lastInstanceClass = dataSet.instance(instanceIterator).stringValue(dataSet.classIndex());
				//System.out.println(lastInstanceClass);
			}
			else
			{
				if(!lastInstanceClass.equals(dataSet.instance(instanceIterator).attribute(dataSet.classIndex()).toString()))
				{
					sameClass = false;
					break;
				}
					
			}
		}
		
		if(sameClass)
		{
			System.out.println("Same class leaf");
			TreeNode node = new TreeNode();
			node.classification = lastInstanceClass.equals("False") ? false: true;
			node.isLeaf = true;
			return node;
		}
		
		// Termination condition 2: We have deleted all other attributes and only classification is left
		if(dataSet.numAttributes()== 1)
		{
			System.out.println("only class leaf");
			int trueCounts = 0, falseCounts = 0;
			trueCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[0];
			falseCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[1];
			TreeNode node = new TreeNode();
			node.classification = falseCounts > trueCounts ? false: true;
			double truePerc = (trueCounts * 1.0)/(trueCounts + falseCounts);
			if(truePerc < 0.6 && truePerc > 0.4)
				node.classification = false;
			node.isLeaf = true;
			return node;
			
		}
		
		// Start ID 3 core algorithm
		// Get the best splitting attribute
		
		
		Util helper = new Util();
		int splittingIndex;
		try {
			splittingIndex = helper.ChooseSplittingAttribute(dataSet, 0.0 );
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
		
		// If no good split was found after chi square test
		if(splittingIndex < 0 || level > 18)
		{
			System.out.println("no good attr leaf");
			int trueCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[0];
			int falseCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[1];
			TreeNode node = new TreeNode();
			node.classification = falseCounts > trueCounts ? false: true;
			double truePerc = (trueCounts * 1.0)/(trueCounts + falseCounts);
			if((truePerc < 0.6 && truePerc > 0.4))
				node.classification = false;
			node.isLeaf = true;
			return node;
		}
		
		
		// separate instances based on attribute value
		
		HashMap<String, Instances> splitDataSet = new HashMap<String, Instances>();
		for(int instanceIter = 0; instanceIter < dataSet.numInstances(); instanceIter++)
		{
			Instance currentInstance = dataSet.instance(instanceIter);
			String valueAtSplittingAttr;
			valueAtSplittingAttr = currentInstance.stringValue(splittingIndex);
			if(currentInstance.isMissing(currentInstance.attribute(splittingIndex)))
				valueAtSplittingAttr = "?";
			
			if (!splitDataSet.containsKey(valueAtSplittingAttr))
			{
				splitDataSet.put(valueAtSplittingAttr, new Instances(dataSet, 0));
			}
			splitDataSet.get(valueAtSplittingAttr).add(currentInstance);
		}
		
		//for each key value, create a child
		System.out.println("Best root is " + dataSet.attribute(splittingIndex).name());
		TreeNode node = new TreeNode();
		node.isLeaf = false;
		node.label= dataSet.attribute(splittingIndex).name();
		for(String attributeValue : splitDataSet.keySet())
		{
			// we don't want to send any values of this attribute
			Instances toPropagate = splitDataSet.get(attributeValue);
			toPropagate.deleteAttributeAt(splittingIndex);	
			node.children.put(attributeValue, BuildTree(toPropagate, level + 1));
		}
	
		return node;
		
	}
	
}
