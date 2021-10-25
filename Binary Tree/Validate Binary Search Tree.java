class Solution {
    public boolean isValidBST(TreeNode root) {
        return searchRight(root.val, root.left) && searchLeft(root.val, root.right);
    }
    
    private boolean searchRight(int val, TreeNode root) {
        
        if (root == null) {
            return true;
        }
        
        if (root.val >= val) {
            return false;
        }
        
        return searchRight(root.val, root.left) && searchLeft(root.val, root.right)&& searchRight(val, root.right);
    }
    
    private boolean searchLeft(int val, TreeNode root) {
        
        if (root == null) {
            return true;
        }
        
        if (root.val <= val) {
            return false;
        }
        
        return searchRight(root.val, root.left) && searchLeft(root.val, root.right) && searchLeft(val, root.left);
    }


    
}