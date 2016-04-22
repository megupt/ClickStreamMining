import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.util.HashMap;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;

public class Util{

	/**
	 * author @meghna
	 */

	public int ChooseSplittingAttribute(Instances data, double confidence) throws Exception
	{
		int attrIndex = InformationGain(data);
		if(attrIndex == -1)
			return -1;
		boolean relevantAttr = CalculateChiSquaredRelevance(data, attrIndex, confidence);
		if (!relevantAttr)
			return -1;
		
		return attrIndex;
		
	}
	
	
	private boolean CalculateChiSquaredRelevance(Instances dataSet, int splittingAttributeIndex, double confidence)
	{
		HashMap<String, Instances> splitDataSet = new HashMap<String, Instances>();
		int trueCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[0];
		int falseCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[1];
		
		double chiSquare = 0;
		for(int instanceIter = 0; instanceIter < dataSet.numInstances(); instanceIter++)
		{
			Instance currentInstance = dataSet.instance(instanceIter);
			String valueAtSplittingAttr = currentInstance.stringValue(splittingAttributeIndex);
			Attribute currentAttr = currentInstance.attribute(splittingAttributeIndex);
			if(currentInstance.isMissing(currentInstance.attribute(splittingAttributeIndex)))
				valueAtSplittingAttr = "?";
			
			if (!splitDataSet.containsKey(valueAtSplittingAttr))
			{
				splitDataSet.put(valueAtSplittingAttr, new Instances(dataSet, 0));
			}
			splitDataSet.get(valueAtSplittingAttr).add(currentInstance);
		}
		int degreesOfFreedom = 0;
		for(String attributeValue : splitDataSet.keySet())
		{
			degreesOfFreedom++;
			Instances currentSet = splitDataSet.get(attributeValue);
			int trueCountsSubSet = currentSet.attributeStats(currentSet.classIndex()).nominalCounts[0];
			int falseCountsSubSet = currentSet.attributeStats(currentSet.classIndex()).nominalCounts[1];
			double expectedTrueSet = (trueCounts * 1.0) * ((trueCountsSubSet + falseCountsSubSet) * 1.0 / (trueCounts + falseCounts));
			double expectedFalseSet = (falseCounts * 1.0) * ((trueCountsSubSet + falseCountsSubSet) * 1.0 / (trueCounts + falseCounts));
		
			if(expectedFalseSet == 0.0 || expectedTrueSet == 0.0)
				return true;
				
			double value = (Math.pow((trueCountsSubSet - expectedTrueSet), 2) / expectedTrueSet) 
					+ (Math.pow((falseCountsSubSet - expectedFalseSet), 2) / expectedFalseSet);
			chiSquare += value;
		}

		ChiSquaredDistribution chiSquareDist = new ChiSquaredDistribution(degreesOfFreedom - 1);
        double invCumProb = chiSquareDist.inverseCumulativeProbability(confidence);
        System.out.println("Chi square test" + chiSquare + " and inv Cum prob " + invCumProb );
        return chiSquare > invCumProb ? true : false;
	}
	
	private static int InformationGain(Instances dataSet)
	{
		int setSize = dataSet.numInstances();
		double setEntropy = Entropy(dataSet);
		double maxInfoGain = Double.MIN_VALUE;
		int chosenAttr = -1;
		for(int attrIter = 0; attrIter < dataSet.numAttributes() - 1; attrIter++)
		{
			HashMap<String, Instances> splitDataSet = new HashMap<String, Instances>();
			for(int instanceIter = 0; instanceIter < dataSet.numInstances(); instanceIter++)
			{
				Instance currentInstance = dataSet.instance(instanceIter);
				Attribute currentAttr = currentInstance.attribute(attrIter);
				String valueAtSplittingAttr = currentInstance.stringValue(currentAttr);
				if(currentInstance.isMissing(currentAttr))
					valueAtSplittingAttr = "?";
				if (!splitDataSet.containsKey(valueAtSplittingAttr))
				{
					Instances data = new Instances(dataSet, 0);
					splitDataSet.put(valueAtSplittingAttr, data);
				}
				splitDataSet.get(valueAtSplittingAttr).add(currentInstance);
			}
			double subSetEntropy = 0;
			for(String attributeValue : splitDataSet.keySet())
			{
				Instances currentSet = splitDataSet.get(attributeValue);
				double entropySet = Entropy(currentSet);
				int currentSetSize = currentSet.numInstances();
				
				subSetEntropy += (((currentSetSize * 1.0)/ setSize) * entropySet);
			}
			
			double currentAttrInfoGain = setEntropy - subSetEntropy;
			
			double intrinsicValue = 0;
			for(String attributeValue : splitDataSet.keySet())
			{
				Instances currentSet = splitDataSet.get(attributeValue);
				int currentSetSize = currentSet.numInstances();
				
				intrinsicValue += (((currentSetSize * 1.0)/ setSize) * (Math.log((currentSetSize * 1.0)/ setSize) / Math.log(2)));
			}
			
			intrinsicValue *= -1;
			double currentAttrGainRatio = (currentAttrInfoGain * 1.0)/ intrinsicValue;
			//System.out.println("Attribute gain for " + attrIter + " is " + currentAttrInfoGain);
			if(currentAttrInfoGain > maxInfoGain)
			{
				maxInfoGain = currentAttrInfoGain;
				chosenAttr = attrIter;
			}
		}
		if(maxInfoGain == 0.0)
			return -1;
			return chosenAttr;
	}
	
	private static double Entropy(Instances dataSet)
	{
		int trueCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[0];
		int falseCounts = dataSet.attributeStats(dataSet.classIndex()).nominalCounts[1];
		
		double posProb = (trueCounts * 1.0) / (trueCounts + falseCounts);
		double negProb = (falseCounts * 1.0) / (trueCounts + falseCounts);
		double entropy = 0;
		if(posProb == 0.0 || negProb == 0.0)
			entropy = 0;
		else
		{
			entropy = -(posProb * (Math.log(posProb)/Math.log(2))) - (negProb * (Math.log(negProb)/Math.log(2)));
		}
		return entropy;
	}
	
}
      
