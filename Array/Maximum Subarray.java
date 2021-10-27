// https://leetcode.com/problems/maximum-subarray/
class Solution {
    public int maxSubArray(int[] nums) {
        
            return recursive(nums,0, nums.length);
        
        
        
        
        
    }
    
    private int recursive(int[] nums, int start, int end) {
    
        if (start == end) {
            return nums[start];
        }
    
        int A = recursive(nums, start, end/2);
        int B = recursive(nums, end/2, end);
        

        if (A > B) {
            return A;
        }

        return B;
        
        
    

}
}