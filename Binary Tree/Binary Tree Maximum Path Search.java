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
        
        return recursive(root)[0];
    }
    
    private int[] recursive(TreeNode root) {
        
        // test if dead end
        if (root.left == null && root.right == null) {
            int[] ret = {root.val, root.val};
            return ret;
        }
        
        int[] L = new int[2];
        if (root.left != null) {
            L = recursive(root.left);
        }


        int[] R = new int[2];
        if (root.right != null) {
            R = recursive(root.right);
        }

        
        int L_max = L[0];
        int L_path = L[1];
        
        
        int R_max = R[0];
        int R_path = R[1];
        
        if (root.left == null) {
            L_max = R_max;
            L_path = 0;
        }
        if (root.right == null) {
            R_max = L_max;
            R_path = 0;
        }
        
        
        int attempt_left = root.val + L_path;
        int attempt_right = root.val + R_path;
        int attempt_none = root.val;
        
        int best = attempt_none;
        if (attempt_left >= attempt_right && attempt_left >= attempt_none) best = attempt_left;
        else if (attempt_right >= attempt_left && attempt_right >= attempt_none) best = attempt_right;
        
        int asParent = root.val + R_path + L_path;
        
        if (best > asParent) asParent = best;
        
        int prop = asParent;
        if (L_max >= R_max && L_max >= asParent) prop = L_max;
        else if (R_max >= L_max && R_max >= asParent) prop = R_max;
        
        
        int[] ret = {prop, best};
        
        return ret;
    }
    
    
   
}