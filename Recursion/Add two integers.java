class Solution {
    public int getSum(int a, int b) {
        return getXOR(a, b);
    }
    
    private int getXOR(int a, int b) {
        int xor = a ^ b;
        int an = (a & b) << 1;
        if (an == 0) {
            return xor;
        }
        return getXOR(xor, an);
    }
}