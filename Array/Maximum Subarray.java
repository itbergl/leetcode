class Solution {
    public int maxSubArray(int[] nums) {
        
        int total = 0;
        int max = -Integer.MIN_VALUE;
        for (int i = 0; i < nums.length; i++) {
            
            int attempt = total + nums[i];
            if (attempt > max) {
                max = attempt;
            }
            if (attempt < 0) {
                total = 0;
            }
            else {
                total += nums[i];
            }  
           
        }
        
        return max;
        
    }
    
}