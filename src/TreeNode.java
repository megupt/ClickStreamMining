import java.util.ArrayList;
import java.util.HashMap;

public class TreeNode {
	
	public String label;
	public boolean isLeaf;
	public boolean classification;
	HashMap<String, TreeNode> children;
	public int labelid;
	
	public TreeNode() {
		isLeaf = false;
		classification = false;
		children = new HashMap<String, TreeNode>();
		label = null;
		labelid = -1;
		
	}

}
