/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */

class Solution {
    public int maxPathSum(TreeNode root) {
        int max = -100000000;
        
        return recursive(root)[0];
    }
    
    
    private int[] recursive(TreeNode root) {
        
        // set branch to 0 if nothing, else recursivly find 
        // path length
        int L = 0;
        int L_parent = root.val -1;
        if (root.left != null) {
            int[] ans = recursive(root.left);
            L = ans[1];
            // if (L < 0 ) L = 0;
            L_parent = ans[0];
        }
        int R = 0;
        int R_parent = root.val -1;
        if (root.right != null) {
            int[] ans = recursive(root.right);
            R = ans[1];
            // if (L < 0 ) L = 0;
            R_parent = ans[0];
        }
        
        // if node is optimal parent node of the best route
        // then update max
        
        int asParent = root.val + L + R;
        if (R_parent >= L_parent && R_parent >= asParent) asParent = R_parent;
        else if (L_parent >= R_parent && L_parent >= asParent) asParent = L_parent;

        // don't consider negative branches
        if (R < 0) R = 0;
        if (L < 0) L = 0;
        
        // find path length of node as a path node
        int asPath = root.val;
        if (R >= L && R >= root.val) asPath += R;
        else if (L >= R && L >= root.val) asPath += L;     
        
        
        int[] info = {asParent, asPath};
        return info;       
      
    }
}