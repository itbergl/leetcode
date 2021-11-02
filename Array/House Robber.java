class Solution {
    public int rob(int[] nums) {
        

        
        int[] vals = new int[nums.length];
        
        for (int i = 0; i < vals.length; i++) {
            vals[i] = -1;
        }
        
        int first = recursion(nums, 0, vals);
        int second = recursion(nums, 1, vals);

        if (first > second) return first;
        return second ;    
    }
        
    
    
    private int recursion(int[] nums, int start, int[] vals) {
        
        if (start >= vals.length) return 0;
        
        if (vals[start]!=-1) return vals[start]; 
        
        int end = nums.length -1;
        
        if (start > end) return 0;
        
        if (end - start < 2) {
            
            int f = nums[start];
            int e = nums[end];
            int ret = 0;
            if (f > e) ret= f;
            else ret = e;
            vals[start] = ret;
            return ret;
        }
        
        int first = recursion(nums, start+3, vals);
        int second = recursion(nums, start+2, vals);
        
        int ret = 0;
        if (first > second) ret= first + nums[start];
        else ret = second + nums[start]; 
        vals[start] = ret;
        return ret;
    }
}