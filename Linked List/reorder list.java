class Solution {
    public void reorderList(ListNode head) {   
   
        HashMap<Integer, ListNode> map = new HashMap<Integer, ListNode>();
             
        // create map
        ListNode node1 = head;
        int L = -1;
        while (node1 != null) {
            L++;
            map.put(L, node1);         
            node1 = node1.next;
        }   
        
        for (int i = 0; i < (L+1)/2 ; i ++) {
            map.get(i).next = map.get(L-i);
            map.get(L-i).next = map.get(i+1);
        }
        map.get((L+1)/2).next = null;      
        return;
         
    }
    

}