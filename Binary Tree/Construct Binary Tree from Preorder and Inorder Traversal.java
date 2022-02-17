/**
 * Definition for a binary tree node.
 * public class TreeNode {
 * int val;
 * TreeNode left;
 * TreeNode right;
 * TreeNode() {}
 * TreeNode(int val) { this.val = val; }
 * TreeNode(int val, TreeNode left, TreeNode right) {
 * this.val = val;
 * this.left = left;
 * this.right = right;
 * }
 * }
 */
class Solution {

    public int index = 0;

    public TreeNode buildTree(int[] preorder, int[] inorder) {

        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();

        for (int i = 0; i < inorder.length; i++) {

            map.put(inorder[i], i);

        }

        return recursive(map, preorder, inorder, 0, preorder.length);
    }

    private TreeNode recursive(HashMap<Integer, Integer> map, int[] preorder, int[] inorder, int ia, int ib) {

        if (index >= preorder.length)
            return null;

        int root = preorder[index];

        if (ib == ia)
            return null;

        Integer i = map.get(root);

        if (i != null) {
            TreeNode newRoot = new TreeNode(root);

            index += 1;

            newRoot.left = recursive(map, preorder, inorder, ia, i);
            newRoot.right = recursive(map, preorder, inorder, i + 1, ib);

            return newRoot;
        }

        return null;

    }

}