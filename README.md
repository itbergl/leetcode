# Amazon Interview Questions

## 1. Valid Parentheses

Solution is to use a stack and add to it when we open a bracket, and remove
the top when it matches a close bracket. If the stack is empty, we have a valid
parentheses. 
- Time - O(n)
- Space - O(n) (using stacks)

```py
def isValid(self, s: str) -> bool:  

    stack = []
    
    close_to_open = {
        ')': '(',
        '}': '{',
        ']': '[',
    }
    
    for c in s:
        if c in close_to_open:

            if len(stack) == 0:
                return False

            if stack.pop() != close_to_open[c]:
                return False
            continue
        
        stack.append(c)

    return len(stack) == 0
```

## 2. Spiral Matrix

Use a direction vector that will rotate clockwise (y,x) -> (x,-y) when it is out of bounds or it will visit a marked cell. When the return array has enough elements, return the result.
- Time O(n*m)
- Space O(n*m)

```py
def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
    
    # direction is of form (y,x)
    rotate_dir = lambda d: (d[1], -d[0])
    
    dirr = (0, 1)
    y, x = 0, 0
    
    size =len(matrix)*len(matrix[0])
    ret = []
    while True:
        ret.append(matrix[y][x])
        matrix[y][x] = '*'
        
        if len(ret) == size:
            break
        
        y_next = y + dirr[0]
        x_next = x + dirr[1]
        
        # test if we need to change direction
        if y_next < 0 or y_next >= len(matrix) or x_next < 0 or x_next >= len(matrix[0]):
            dirr = rotate_dir(dirr) 
        elif matrix[y_next][x_next] == '*':
            dirr = rotate_dir(dirr)
        
        # new position
        y +=  dirr[0]
        x +=  dirr[1]
        
    return ret
```

## 3. Valid Anagram

Create a dictionary of frequencies of each letter in one of the words. Iterate through the second, subtracting from the frequency dictionary. After, if the dictionary contains any non 0 frequency then there is no valid anagram.

```py
def isAnagram(self, s: str, t: str) -> bool:
    
    if len(s) != len(t):
        return False
    
    freq_s = {}
    
    for c in s:
        freq_s[c] = freq_s.get(c, 0) + 1
    
    for c in t:
        if c not in freq_s:
            return False
        
        freq_s[c] -= 1
    
    for c in freq_s:
        if freq_s[c] != 0:
            return False
        
    return True
```

## 4. Set Matrix Zeroes

Iterate through each cell and if it is a 0, then add colum and row to a list. Remove duplicates from the lists after by casting to a set. For each row, replace with a fixed length array of 0's, for each column, replace each first entry with 0's.

```py
def setZeroes(self, matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """
    
    rows = []
    columns = []
    
    for r in range(len(matrix)):
        for c in range(len(matrix[r])):
            
            if matrix[r][c] == 0:
                rows.append(r)
                columns.append(c)
    
    rows = set(rows)
    columns = set(columns)
    
    for r in rows:
        matrix[r] = [0]*len(matrix[0])
    
    for c in columns:
        for r in range(len(matrix)):
            matrix[r][c] = 0
    
    return matrix
```

## 5. Lowest Common Anscestor of BST

Do a binary search on the BST until you find a value that is between smallest number and largest number. Keep iterating, aiming for not a value but the range.

- Time O(logn)
- Space O(1)

```py
def dfs(self, node, small, big):
    if node.val >= small and node.val <= big:
        return node
    
    if node.val > big:
        if node.left:
            return self.dfs(node.left, small, big)
    elif node.val < small:
        if node.right:
            return self.dfs(node.right, small, big)

    return None
    
    
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

    return self.dfs(root, min(p.val,q.val), max(p.val, q.val))
```

## 6. Construct Binary Tree From preorder and inorder traversal

Recursively pop leftmost element from preorder, as we know that will be the topmost head. Find the index in inorder of the popped element (optimised as a hashmap). We can split the inorder into a left tree (before index) and a right tree (after index). If there are elements in the left tree, they will appear before the right children in the preorder list so we can pop from the left again and use the left subtree recursively. 

Infact, we don't even need to pass in inorder, as we have already cached the indexes, and the only use of tracking the left and right children is to determine if an element in preorder doesn't belong to a left child of a node (for example the l subtree from the index may be empty so we would know to skip it).

- Time O(n)
- Space O(n)

```py
def build(self, root, index, preorder, p_i, inorder, l, r):
    if not preorder:
        return 
            
    root.val = preorder[p_i[0]]
    p_i[0] += 1

    i = index[root.val]
    if l < i:
        root.left = TreeNode()
        self.build(root.left, index, preorder, p_i, inorder, l, i)
        
    if i+1 < r:
        root.right = TreeNode()
        self.build(root.right, index, preorder, p_i, inorder, i+1, r)


def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    index = {v:i for i, v in enumerate(inorder)}
    root = TreeNode()
    self.build(root, index, preorder, [0], inorder, 0, len(inorder))
    return root
```

## 7. Counting Bits

To do in nlogn time, we could either keep a complicated datastructure that keeps track of each power of two column, or decompose each element into its powers of two. To do in O(n) time in a single pass we can find our largest power of two found so far, knowing the solution is 1 for this number, and finding the difference between the subsequent numbers. This difference will already be cached so we can simply add the solutions. We know that the subsequent numbers are equal to the previous numbers but with an additional 1 at the front so we can just append the differnce that is already cached + 1.

- Time O(n)
- Space O(n) (only because solution is an array of size n)

```py
def countBits(self, n: int) -> List[int]:
    ret = [0]
    max_pow = 1
    count = 0
    for i in range(1, n+1):
        
        if i == 2*max_pow:
            max_pow = i
            ret.append(1)
            continue
            
        sub = i - max_pow
        ret.append(1 + ret[sub])
    
    return ret
        
```

## 8. Two Sum

Keep a list of elements that we need to find, when we come across a number. For example if the target is 5 and we find 3, store 2 and the index. If our element is not beening looked for add its pair to the list and continue.
- Time O(n)
- Space O(n)

```py
def twoSum(self, nums: List[int], target: int) -> List[int]:
    
    seen = {}
    for i, n in enumerate(nums):
        if target-n in seen:
            return [i, seen[target-n]]
        
        seen[n] = i
    
    return
```

## 9. Non-overlapping Intervals

Firstly, we note that the first interval that ends first must be included in the solution. If we consider any comnbination of intervals that could be the optimal answer (maximum intervals included) we could always either add on the first ending interval, or replace the current first interval with it as worse case it would be a one for one replacement. So we sort the intervals by ending time, and keep track of the previous end. We iterate through the sorted intervals, if it is non overlapping we don't include, otherwise increment a counter. We then return the number of total intervals minus this count.
- Time O(nlogn + n), nlogn for sort, n for iterating intervals
- Space O(logn), for quicksort (logn call stacks)

```py
def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
    
    # sort by start time
    intervals.sort(key=lambda x: x[1])

    prev_end = intervals[0][0]
    
    count = 0

    for i in intervals:     
        if i[0] >= prev_end:
            count += 1
            prev_end = i[1]

    
    return len(intervals)-count
```

## 10. Best Time to buy and sell stock

My solution is as follows: iterate through the array forward, keeping track of the smallest value seen so far, and at each stage calculating the profit and updating max profit appropriately.
- Time O(n)
- Space O(1)

```py
def maxProfit(self, prices: List[int]) -> int:
    
    lowest_so_far = 10000
    ans = -1
    for p in prices:
        lowest_so_far = min(p, lowest_so_far)
        ans = max(p - lowest_so_far, ans)
        
    return ans
```

While I think my solution is more readable, there are some unessessary comparisons. For example, we are always checking min value seen when really we shouldn't have to. 

Another solution is to use two pointers. L = buy, R = sell, initially 0, 1. We will march R forward until it is less than the value pointed to by L, checking max profit at all stages. When it is less than L, set L to be where right is, and march R forward. Basically, we are keeping a pointer at the min value seen at all times, and checking every sell price until a new min value is found.
- Time O(n)
- Space O(1)

```py
def maxProfit(self,prices):
    left = 0 #Buy
    right = 1 #Sell
    max_profit = 0
    while right < len(prices):
        currentProfit = prices[right] - prices[left] #our current Profit
        if prices[left] < prices[right]:
            max_profit =max(currentProfit,max_profit)
        else:
            left = right
        right += 1
    return max_profit
```

## 11. Climbing Stairs

With DP, we can build an array of size n where the ith entry is the solution for n=i. To find the solution for n=n+2, we can add n and n+1 as there is one way to get to n+2 from n and one way from n+1 in one step. A further optimisation is that we only care about the last two elements so we can just use a sliding window of size 2.
Time - O(n)
Space - O(1)

```py
def climbStairs(self, n: int) -> int:
    
    if n == 1:
        return 1
    
    window = (1, 2)
    
    for i in range(n-2):
        
        i, j = window
        window = j, j + i 
    
    return window[1]
```

## 12. Binary Tree Level Order Traversal

I had originally tried to be too clever and put a hashmap where it had no buisiness being. I knew to do a BFS as this is exactly a level order traversal. I had a hashmap of levels to elements, with the queue of the BFS including an incremented level variable as a two tuple with the node. Then I used a list comprehension to append these lists. This was very slow (beat 5% by time).

Instead I built the return list as I went. I kept the idea of attaching level variable to each queue element. Now I just append to the last element unless there is a new level.

Time - O(n)
Space - O(n), as solution is an array

```py
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    
    if not root:
        return []
    
    levels = []
    
    queue = collections.deque([(root, 0)])
    
    while queue:
        
        n, l = queue.popleft()
        
        if l > len(levels)-1:
            levels.append([n.val])
        else:
            levels[l].append(n.val)
        
        if n.left:
            queue.append((n.left, l+1))
        if n.right:
            queue.append((n.right, l+1))
    
    return levels
```

## 13. Pacific Atlantic Water Flow

My solution, which I think is very readable and runs in the same time, is to do a DFS from the inner corner of each ocean, finding a set of all possible paths. Then we just find the common elements, which can be done with sets intersection.
- Time O(n*m)
- Space O(n*m)

```py
def dfs(self, heights, seen, r, c, edges):
    seen.add((r,c))
    for dr, dc in self.directions:
        if 0 <=  r+dr < len(heights) and 0 <= c+dc < len(heights[0]) and (r+dr,c+dc) not in seen:
            if r+dr == edges[0] or c+dc == edges[1] or heights[r+dr][c+dc] >= heights[r][c]:   
                self.dfs(heights, seen, r+dr, c+dc, edges)


def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    self.directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    p_seen = set()
    self.dfs(heights, p_seen, 0, 0, (0,0))
    
    a_seen = set()
    self.dfs(heights, a_seen, len(heights)-1, len(heights[0])-1, (len(heights)-1, len(heights[0])-1))
    return list(p_seen & a_seen)
```

## 14. Longest Consecutive Sequence

This one was annoying. First I constructed a mapping of every element index to the index of the number below it (if present). Next I iterated through that graph, and when I hit the bottom of a sequence I remembered the depth. When I did further iterating and hit an already seen sequence head I would add the solutions and store in max.

- Time O(n)
- Space O(n)

```py
def longestConsecutive(self, nums: List[int]) -> int:
    
    if not nums:
        return 0
    
    nums_set = {n:i for i,n in enumerate(set(nums))}
    
    decending = {}
    
    for n, i in nums_set.items():
        if n+1 in nums_set:
            decending[nums_set[n+1]] = i

    longest_decending = {}
    ans = 1
    for i in decending:
        count = 1
        f = i
        while f in decending and f not in longest_decending:
            count += 1
            f = decending[f]
        
        if f in longest_decending:
            count += longest_decending[f]-1
        
        ans = max(ans, count)
        longest_decending[i] = count
        
        
    return ans
```

However, this solution is pretty bad for memory. Hence we can simply make a set of the numbers, and for every number in the sequence start searching up only if it is the bottom of a sequence (n-1 not in set).
- Time O(n)
- Space O(n)

```py
def longestConsecutive(self, nums):
longest_streak = 0
num_set = set(nums)

for num in num_set:
    if num - 1 not in num_set:
        current_num = num
        current_streak = 1

        while current_num + 1 in num_set:
            current_num += 1
            current_streak += 1

        longest_streak = max(longest_streak, current_streak)

return longest_streak
```

## 15. Reverse Linked List

Keep two variables, and attach the front to behind and increment
- Time O(n)
- Space O(1)
```py
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    
    if not head:
        return
    
    a, b = None, head
    
    while b:
        temp = b.next
        b.next = a
        a, b = b, temp
    return a
```

## 16. Reorder List

Another frustrating one. One solution was to make a deque of all the elements in order, and then swap between popping from the left and then right. Another solution that I implemented below is to find the middle and use a complicated recursive function that makes a swap and returns the right outer node.
- Time O(n)
- Space O(n) (technically because of recursive calls)
  
```py
def swap(self, node, count, length):  
    if count == length//2-1:
        ret = node.next.next
        node.next.next = None
        
        if length%2 == 0:
            return ret
        else:
            end = ret
    else:
        end = self.swap(node.next, count+1, length)
        
    ret = end.next     
    node.next, end.next = end, node.next
    return ret


def reorderList(self, head: Optional[ListNode]) -> None:
    """
    Do not return anything, modify head in-place instead.
    """
    if not head:
        return
    if not head.next:
        return 
    
    front = head
    length = 0
    while front:
        length += 1
        front = front.next
        
    count = 0
    self.swap(head, count, length)
```

However, due to the O(n) nature of the recursion, this is bad for memory (but great for time).

The solution I was "meant" to see involves splitting the list down the middle and reversing the end segement and then inserting the lists into eachother. I had realised the importance of the middle element in my solution, but didn't see this as a linked list insertion. Perhaps if I had done that problem recently I could have seen it.
- Time O(n)
- Space O(1)

```py
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        
        slow = head
        fast = head
        mid = 0
        while fast and fast.next:
            mid += 1
            fast = fast.next.next
            slow = slow.next
        
        
        a, b = None, slow.next
        slow.next = None
        while b:
            temp = b.next
            b.next = a
            a, b = b, temp
        
        
        A = head
        B = a
        
        while B:
            temp = B.next
            A.next, B.next = B, A.next
            A = B.next
            B = temp
```



## 17. Longest Repeating Character Replacement

I hated this one so much. I tried doing a two pointer type thing, which was close but no cigar. The actual solution is to keep a frequency map of the elements within the window, and find the max frequency to obtain how many characters need to be changed to fit the window. Based on this you can grow or shrink the window. Also, because we only care when we get a new max frequency, we can just keep track of that.
- Time O(n)
- Space O(n)

```py
def characterReplacement(self, s: str, k: int) -> int:

    l, r = 0,0
    
    window_frequency = collections.defaultdict(int)
    window_frequency[s[l]] += 1
    max_freq = 1
    ans = k
    while r < len(s):
        if 1+ r-l - max_freq > k:
            window_frequency[s[l]] -= 1
            l += 1
        else:
            ans = max(ans, 1+r-l)
            r += 1
            if r < len(s):
                window_frequency[s[r]] += 1
                max_freq = max(max_freq, window_frequency[s[r]])
    return ans  
```

## 18. Binary Tree Maximum Path Sum

I think I have done this one before. Basically you use a recursive function that returns the maximum path that has the input node as a root, which is easy enough at the bottom of the tree. Every time you return the best tree. Along the way you can compare what the combination of the left and right paths would be and update a global variable. 
- Time O(n)
- Space O(n) (recursion)
  
```py
def dfs(self, node, ans):
    
    if not node:
        return 0
    
    mps_l_root = self.dfs(node.left, ans)
    mps_r_root = self.dfs(node.right, ans)
    
    mps_l_root = max(mps_l_root, 0)
    mps_r_root = max(mps_r_root, 0)
    
    as_root = max(mps_l_root, mps_r_root)+node.val
    ans[0] = max(ans[0], mps_l_root + mps_r_root + node.val)
    return as_root


def maxPathSum(self, root: Optional[TreeNode]) -> int:
    
    ans = [-1000]
    
    self.dfs(root, ans)
    return ans[0]
```

## 19. Reverse Bits

This is something I'm not at all familiar with. The solution is to iterate 32 times, each time pushing the answer right by one bit, exposing a new bit on the right, and add the right bit of the input number (by bitwise AND). Then we shift the input right one bit to get the next bit.
- Time O(1)
- Space O(1)
  
```py 
def reverseBits(self, n: int) -> int:
    
    ans = 0
    for _ in range(32):
        
        ans = (ans << 1) + (n & 1)
        n >>= 1
    
    return ans
```

## 20. Minimum Window Substring

This question is the bane of my existance. The solution is to keep a frequency counter of t, and subtract from it whenever the window meets a character in t. When the frequencies are positive, this means we haven't found our first match, so decrement a counter of size len(t) until it reaches 0 at which point we have our first match. 

Next increase the left pointer, adding to the frequencies as you go. If you add to the frequency and it becomes 0 that means we have used up all our reserves for that character. We can record the optimal answer and move the left pointer along - resetting the counter to 1 (as we know we just disregarded a t letter) and increasing the frequency.

- Time O(n)
- Space O(t)

```py
def minWindow(self, s: str, t: str) -> str:
    
    freq = collections.Counter(t)
    
    missing = len(t)
        
    best = ''
    length = len(s)+1
    
    i = 0
    
    for r, c in enumerate(s, 1):
        
        if freq[c] > 0:
            missing -= 1
        freq[c] -= 1
        
        if missing == 0:
            
            while i < r and freq[s[i]] < 0:
                
                freq[s[i]] += 1
                i += 1
            
            missing = 1
            freq[s[i]] += 1
            
            if r-i < length:
                length = r-i
                best = s[i:r]
            i += 1
    return best
```

## 21. Longest Substring Without Repeating Characters

Keep a dictionary of the last time we've encountered this character. If it is in our current window when we find it again then swap the start of the window to just after it. When we update the right pointer, record the last seen index for the current character and update the max value.
- Time O(n)
- Space O(n)

```py
def lengthOfLongestSubstring(self, s: str) -> int:

    last_seen = {}
    i = 0
    best = 0
    for j, c in enumerate(s):
        
        if last_seen.get(c, -1) >= i:
            
            i = last_seen[c] + 1
        else:
            best = max(best, j-i+1)

        last_seen[c] = j 
        

    return best
```

## 22. Implement Trie

Insert - for each character, place in the dictionary and progress the tree with a new dictionary if there is no key.
Search - for each charactter, iterate the trie and see if the end of word character exists
Prefix - same as search but don't check end-of-word character

Insert
- Time O(n)
- Space O(n)

Search
- Time O(n)
- Space O(1)

Prefix
- Time O(n)
- Space O(1)

```py
def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        
        head = self.root
        for c in word:
            head[c] = head.get(c, {})
            head = head[c]
        head['.'] = ''
           

    def search(self, word: str) -> bool:
        head = self.root
        for c in word:
            if c not in head:
                return False
            head = head[c]
        return '.' in head
            
            
    def startsWith(self, prefix: str) -> bool:
        head = self.root
        for c in prefix:
            if c not in head:
                return False
            head = head[c]
        return True
```

## 23. Valid Palindrom

Process the input to only include alphanumerics and convert to lowercase. Compare two enclosing pointers until they meet in the middle.

- Time O(n)
- Space O(1)
  
```py
def isPalindrome(self, s: str) -> bool:
    aplhabet = set('abcdefghijklmnopqrstuvwxyz0123456789')
    s = [c for c in s.lower() if c in aplhabet]
    
    i, j = 0, len(s)-1
    while i <= j:
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    return True
```

## 24. 3Sum

## 25. Merge Two Sorted Lists

Create a bottom and top pointer, where the bottom is where we are inserting into and is initially a smaller value than the top. Keep iterating until the next node is bigger than the top pointer and then make the switch.

- Time O(n)
- Space O(1)

```py
def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:

    if not list1:
        return list2

    if not list2:
        return list1


    top = list1
    bottom = list2
    if list1.val < list2.val:
        top = list2
        bottom = list1

    head = bottom

    while top:

        if not bottom.next:
            bottom.next = top
            break

        if bottom.next.val < top.val:
            bottom = bottom.next
        else:
            bottom.next, top.next, top = top, bottom.next, top.next
            bottom = bottom.next

    return head
```

## 26. Course Scheduling 

Do a DFS (topological sort) on a directed graph of course dependencies. Keep a set of nodes in the current path and a set of explored nodes. If a node in the dfs is in the path already, return True for the presenence of a cycle. If it is already explored then we can ignore it.

- Time O(n)
- Space O(n)

```py
def dfs(self, node, adj_list, explored, path):

    if node in path:
        return True

    path.add(node)

    for neighbour in adj_list.get(node, []):
        if neighbour in explored:
            continue 
        if self.dfs(neighbour, adj_list, explored, path):
            return True

    explored.add(node)
    path.remove(node)
    return False


def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:

    adj_list = {}

    for p in prerequisites:

        adj_list[p[1]] = adj_list.get(p[1], []) + [p[0]]

    path = set()
    explored = set()

    for node in adj_list:
        if self.dfs(node, adj_list, explored, path):
            return False

    return True
```

## 27. Number of Islands

My solution was to simply edit the matrix inplace and do a dfs at each node. The number of dfs calls is the number of islands:
- Time O(n*m)
- Space O(1)

```py
def dfs(self, r, c, grid):


    grid[r][c] = '2'
    for dr, dc in [(0,1), (0,-1), (-1,0), (1,0)]:
        if 0<=r+dr<len(grid) and 0<=c+dc<len(grid[0]):
            if grid[r+dr][c+dc] == '1':
                self.dfs(r+dr, c+dc, grid)



def numIslands(self, grid: List[List[str]]) -> int:
    count = 0
    for r in range(len(grid)):
        for c in range(len(grid[0])):

            if grid[r][c] =='1':
                count += 1
                self.dfs(r,c,grid)
    return count
```

## 28. 

## 29. Rotate Image

Flip along the line r=c and then the line c = mid, where mid is the midpoint of n.

- Time O(n^2)
- Space O(1)

```py
def rotate(self, matrix: List[List[int]]) -> None:
    """
    Do not return anything, modify matrix in-place instead.
    """

    n = len(matrix)
    # reflect along r = c

    for r in range(n):
        for c in range(r, n):
            matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]


    # reflect along c = mid

    for r in range(n):
        for c in range(n//2):
            matrix[r][c], matrix[r][n-c-1] = matrix[r][n-c-1], matrix[r][c]
```

## 31. Merge K Sorted Lists

I have ommitted the code for merge 2 sorted lists as I have already solved that problem. This problem uses that solution and basically merges the lists together heirachically. I'm sure there are optimisations to be made (such as optimally merging the lists to keep equal sizes), but this solution if fine. 
- Time O(n*average_list_length)
- Space O(n)

```py
def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

    k = len(lists)

    if k == 0:
        return None
    if k == 1:
        return lists[0]

    while len(lists) > 1:

        lists = [lists[i] if i==len(lists)-1 else self.merge2Lists(lists[i], lists[i+1]) for i in range(0, len(lists), 2)]
        # swap to ensure last element isn't always ignored
        lists[0], lists[-1] = lists[-1], lists[0]


    return lists[0]
```

## 32. Word Break

My solution was to use a dfs with caching. 
- Time O(n)
- Space O(n)

```py
def dfs(self, wordDict, s, i, seen):

    if i == len(s):
        return True

    for w in wordDict:

        if s.startswith(w, i) and not i in seen:

            if self.dfs(wordDict, s, i+len(w), seen):
                return True
            if i+len(w)==len(s):
                return True

    seen.add(i) 
    return False


def wordBreak(self, s: str, wordDict: List[str]) -> bool:

    seen = set()

    return self.dfs(wordDict, s, 0, seen)
```

However, top-down memoisation could also be used:
- Time O(n)
- Space O(n)

```py
def wordBreak(self, s: str, wordDict: List[str]) -> bool:

    memo = [False]*(1+len(s))
    memo[0] = True

    for i in range(1, len(s)+1):

        for w in wordDict:

            if i >= len(w) and s[i-len(w):i] == w:

                if memo[i]:
                    continue
                memo[i] = memo[i-len(w)]

    return memo[-1]
```
