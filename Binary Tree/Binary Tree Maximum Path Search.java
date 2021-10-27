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
        
        recursive(root, max);
    }
    
    
    private int[] recursive(TreeNode root, int max) {
        
        // set branch to 0 if nothing, else recursivly find 
        // path length
        int L = 0;
        if (root.left != null) {
            L = recursive(root.left, max);
            
        }
        int R = 0;
        if (root.right != null) {
            R = recursive(root.right, max);
            
        }
        
        // if node is optimal parent node of the best route
        // then update max
        if (root.val + L + R > max) {
            System.out.println(max);
            max = root.val + L + R ;
        }
        // don't consider negative branches
        if (R < 0) R = 0;
        if (L < 0) L = 0;
        
        // find path length of node as a path node
        int ret = root.val;
        if (R >= L && R >= root.val) ret+= R;
        else if (L >= R && L >= root.val) ret += L;     
        
        return ret;       
      
    }
}