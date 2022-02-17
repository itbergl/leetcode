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

    public List<List<Integer>> levelOrder(TreeNode root) {

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        Queue<Integer> level = new LinkedList<Integer>();

        List<List<Integer>> ret = new ArrayList<List<Integer>>();

        if (root == null)
            return new ArrayList<List<Integer>>();

        queue.add(root);
        level.add(0);

        while (!queue.isEmpty()) {

            TreeNode el = queue.poll();
            Integer l = level.poll();

            if (l == ret.size()) {
                ret.add(new ArrayList<Integer>());
            }

            ret.get(l).add(el.val);

            if (el.left != null) {
                queue.add(el.left);
                level.add(l + 1);
            }

            if (el.right != null) {
                queue.add(el.right);
                level.add(l + 1);
            }

        }

        return ret;

    }

}