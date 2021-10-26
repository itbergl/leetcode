class Solution {
    public int[] twoSum(int[] nums, int target) {
        return findSum(nums, target);
    }
    
    private int[] findSum(int[] nums, int target) {
        
        HashMap<Integer, Integer> found = new HashMap<Integer, Integer>();
        
        for (int i = 0; i < nums.length; i++) {
            
            int a = nums[i];
                      
            Integer key = found.get(target - a);
            
            if (key != null) {
                int[] ret = {i, key};
                return ret;
            }
            else {
                found.put(a, i);
            }
            
        }
        // never reached
        int[] ret = {0, 0};
        return ret;
        
        
        
    }
    
    
}