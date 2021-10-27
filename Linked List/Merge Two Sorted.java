class Solution {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }

        if (l1.val < l2.val) {
            recursive(l1, l2);
            return l1;
        }
        else {
            recursive(l2, l1);
            return l2;
        }
        
    }
    
    
    private void recursive(ListNode l1, ListNode l2) {  
        
        if (l1.next != null) {
            int next = l1.next.val;
            int other = l2.val;
            if (next < other) {
                recursive(l1.next, l2);
            }

            else {
                ListNode detached = l1.next;
                l1.next = l2;
                recursive(l2, detached);

            }
        }
        else {
            l1.next = l2;
            return;
        }
        
    }
}