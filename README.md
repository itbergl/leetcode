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

Sort the list O(nlogn), and point to the left index. Make a pointer to the right and at the end. If the sum is above 0 move the right and if it is less than 0 move the left. When it is 0 save the answer. 

- Time O(n^2)
- Space O(1)
```py
def threeSum(self, nums):
    # i + j + k = 0
    ret = []
    nums = sorted(nums)
    prev = nums[0]
    for i in range(len(nums[:-2])):
            
        if nums[i] == prev and i > 0:
            continue
        prev = nums[i]
        
        j, k = i+1, len(nums)-1

        while j < k:

            curr_sum = nums[j] + nums[k]

            if curr_sum > -nums[i]:
                k -= 1
            elif curr_sum < -nums[i]:
                j += 1
            else:
                ret.append([nums[i], nums[j], nums[k]])
                found = nums[j]
                while j < k and nums[j] == found:
                    j += 1
    return ret
```

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

## 28. Longest Common Subsequence

This has got to be the hardest problem I have done. I simply do not understand dynamic programming. I had tried solving with cached DFS, which took too long. 

The solution is from the 2-DP relation with DP(a,b) being the solution for the subproblem with strings A\[:a\] and B\[:b\]. 

- If we consider DP(a+1, b+1), the solution will be 1+DP(a,b) if A\[a+1\] == B\[b+1\]. We can introduce this as a rule. 
- If we consider DP(a+1, b+1), and A\[a+1\] != B\[b+1\], the solution will the max of the solutions before we added a new character: ``max(DP(a-1, b), DP(a, b-1)``. Simply, because adding the end characters does not create a better solution, we know the best possible solution will be equivalent to the best solution found so far, which is non-decreasing. That solution was either found when we neglected either of the end characters on both strings.

- Time O($n*m$)
- Space O($n*m$)

```py
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    
    dp = [[0 for _ in range(len(text1) + 1)] for _ in range(len(text2)+1)]
    
    for i in range(1, len(text1)+1):
        
        for j in range(1, len(text2)+1):
            topleft = dp[j-1][i-1]
              
            if text1[i-1] == text2[j-1]:
                dp[j][i] = topleft + 1
                continue

            up, left = dp[j][i-1], dp[j-1][i]
            dp[j][i] = max(up, left)

    return dp[-1][-1]        
```

An obvious optimisation is to only use 2 rows of space, and index by modulo or otherwise.

- Time O($n*m$)
- Space O($n$)

```py
def longestCommonSubsequence(self, text1: str, text2: str) -> int:
    
    dp = [[0 for _ in range(len(text1) + 1)] for _ in range(2)] 
         
    for i in range(1, len(text2)+1):  
        for j in range(1, len(text1)+1):
            
            up = dp[(i+1) & 1][j]
            left = dp[i & 1][j-1]
            topleft = dp[(i+1)& 1][j-1]
            
            if text2[i-1] == text1[j-1]:
                dp[i& 1][j] = topleft + 1
            elif up >= left:
                dp[i& 1][j] = up
            else:
                dp[i& 1][j] = left
    return dp[len(text2)& 1][-1]
```
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

## 33. Combination Sum

A basic solution for this that holds up pretty well in regards to time and space complexity is to do a simple DFS with caching.

- Time O(n Choose n/N)?
- Space O(n Choose n/N)?

```py
def dfs(self, path, candidates, target, solutions, seen):
        
        path_tuple = tuple(sorted(path))
        if (target, path_tuple) in seen:
            return
        
        seen.add((target, path_tuple))
        
        for c in candidates:
            
            if target -c == 0:
                solutions.add(tuple(sorted(path + [c])))
                continue
            
            if target-c < 0:
                continue
                
            self.dfs(path + [c], candidates, target-c, solutions, seen)
        
        
    
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        ret = set()
        seen = set()
        self.dfs([], candidates, target, ret, seen)
        return list(ret)     
```


## 34. Unique Paths

To solve this problem, you can do 2D dynamic programming.

- Time O(n*m)
- Space O(n*m)

```py
def uniquePaths(self, m: int, n: int) -> int:
    
    dp = [[1]*m for _ in range(n)]
    
    for l in range(m):
        for d in range(n):
            top = dp[d-1][l] if d > 0 else 0
            left = dp[d][l-1] if l > 0 else 0
            dp[d][l] = top + left
    
    return dp[-1][-1]
```

However, since we are only looking at one row of the 2D array at a time, we can do this in O(n) space

- Time O(n*m)
- Space O(n)

```py
def uniquePaths(self, m: int, n: int) -> int:
    
    dp = [1]*n
    for _, j in product(range(1, m), range(1, n)):
        dp[j] += dp[j-1]
    return dp[-1]
```

## 35. Decode Ways

DP solution. Create an array of size n that denotes, at i, the solution for the substring s\[:i\]. After checking that the first element isn't 0, we know the first solution is 1. Start indexing from the second element, and if that element isn't 0 our next solution will be the previous one. If, additionally, the previous two elements are less than 27 and greater than 9 we should add those paths.

- Time O(n)
- Space O(n)

And to reduce memory usage, have a sliding window of size two.

- Time O(n)
- Space O(1)
  
```py
def numDecodings(self, s: str) -> int:
    
    
    if int(s[0]) == 0:
        return 0
    
    dp = (1,1)
    
    
    
    for i, c in enumerate(s[1:], 1):
        
        one = int(c)
        n = 0
        if one != 0:
            n += dp[1] 
        
        
        two = int(s[i-1:i+1])

        if two < 27 and two >= 10:
            n += dp[0] 
        
        dp = (dp[1], n)
        
    return dp[-1]
```

## 36. Remove Nth Node From End Of List

I had the right intentions for this one, but the wrong execution. Create a fast and slow pointer and increment the fast one by ``n``. Then start incrementing both until the end is reached. I had originally used a counter instead of a finite for loop, assigned head to slow only after fast had been reached and made a redundant check for edge case of a single node input but otherwise had the right idea.

- Time O(n)
- Space O(1)

```py
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    
    fast, slow = head, head
    for _ in range(n): 
        fast = fast.next
        
    if not fast: 
        return head.next
    
    while fast.next: 
        fast, slow = fast.next, slow.next
        
    slow.next = slow.next.next
    return head
```

## 37. Add and Search Word

Implement a Trie - a tree of prefixes. Recursively search wildcards to find matches.

### ``__init__()``
- Time O(1)
- Space O(1)

## ``addWord(word)``

- Time O(len(word))
- Space O(len(word))


## ``search(word)``

- Time O(26^w), where w is the number of wildcards and 26 is the number of letters in the alphabet
- Space O(1)

```py
def __init__(self):
    self.trie = dict()

def addWord(self, word: str) -> None:
    
    head = self.trie
    for c in word:
        if c not in head:
            head[c] = dict()
        head = head[c]
    head['_'] = dict()
    
def search_dfs(self, word, trie):
    if not word:
        return '_' in trie
    
    for i, c in enumerate(word):
        if c == '.':
            return any(self.search_dfs(word[i+1:], trie[g]) for g in trie)  
        elif c not in trie:
            return False
        trie = trie[c]
    return '_' in trie
        
def search(self, word: str) -> bool:
    return self.search_dfs(word, self.trie)
```

## 38. Invert/Flip Binary Tree

Classic

- Time O(n)
- Space O(n) (callstack)

```py
def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
    
    if not root:
        return root
    
    self.invertTree(root.left)
    self.invertTree(root.right)
    
    root.left, root.right = root.right, root.left
    return root
```

## 39. Product of Array Except Self

Keep an array of the product to the left at each index. Keep an array of the product to the right at each index. Interate through and append to an array.

Instead of appending to an array for the output, multiply in-place while constructing the "to-the-right" array:

- Time O(2n)
- Space O(2n)

```py
def productExceptSelf(self, nums: List[int]) -> List[int]:
    
    ret, left = [], []
    last = 1
    for i, n in enumerate(nums):
        left.append(last)
        ret.append(last)
        last = n*last
        
    last = 1
    for i, n in enumerate(nums[::-1]):
        ret[-(i+1)] *= last
        last = n*last
    
    return ret
```

More concisely, you can modify the "to-the-left" array in-place for O(n) space, but it looks uglier.

- Time O(2n)
- Space O(n)
  
```py
def productExceptSelf(self, nums: List[int]) -> List[int]: 
    ret = [1]
    for n in nums[:-1]:
        ret.append(ret[-1]*n)
    
    temp = 1
    for i, n in enumerate(nums[::-1]):
        ret[-(i+1)] *= temp
        temp *= n
    return ret     
```

## 40. Maximum Subarray

Iterate through the array and add the running total. If the total becomes negative, it is a useless prefix and we can set the total to 0. At every iteration we update the maximum.

- Time O(n)
- Space O(1)
```py
def maxSubArray(self, nums: List[int]) -> int:
    total, max_val = 0, float('-inf')

    for n in nums:          
        attempt = total + n
        if attempt > max_val:
            max_val = attempt
        
        if attempt < 0:
            total = 0
        
        else:
            total += n

    return max_val
```

## 41. Coin Change

Make array of length amount+1, where the index represents the solution for amount = index. At each index consider each coin denomination and look ahead to that solution. Replace the value with the min of itself and the current index + 1. 

- Time O(n)
- Space O(n)

```py
def coinChange(self, coins: List[int], amount: int) -> int:
    
    dp = [amount+1]*(amount+1)
    dp[0] = 0
    
    for a in range(amount):          
        for c in coins:
            if a+c <= amount and dp[a]+1 < dp[a+c]:
                dp[a+c] = dp[a]+1
    
    return dp[-1] if dp[-1] != amount+1 else -1
```

## 42. Kth Smallest Element in a BST 

### Iterative Solution

Keep a stack of the sorted traversal by walking down the left tree, until you hit null. Subtract one from k and if its 0 then you have an answer, else point to the right node. If you don't have a left path to traverse then the stack will have the next smallest value at the top.

- Time O(n)
- Space O(log(n))
```py
def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    
    stack = deque()
    curr = root
    
    while curr or stack:
        
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        k -= 1
        
        if k == 0:
            return curr.val
        
        curr = curr.right
```

### Recursive Solution

Search left tree, and if it is null return -1. If the left tree is -1 then subtract 1 from the global counter. If the counter is 0 then the node itself is the return value, but if not return the answer from the right tree.

- Time O(n)
- Space O(log(n))

```py
class Solution:
    def dfs(self, node):
            
        if not node:
            return -1

        left = self.dfs(node.left)

        if left != -1:
            return left

        self.k -= 1

        if self.k == 0:
            return node.val

        return self.dfs(node.right)
    
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        return self.dfs(root)
```

To improve, include number of left children in each node.

## 43. Missing Number

### Math Solution

Since we are missing exactly one number, then we could take the sum of all the numbers there should be, subtract the numbers that are there.

- Time O(n)
- Space O(1)

```py
def missingNumber(self, nums: List[int]) -> int:
    
    return int((len(nums) * (len(nums) + 1) )/ 2 -  sum(nums))
```

## Loop Solution

Sort the array and go through until one of the values isn't one greater than the previous value.
- Time O(nlogn)
- Space O(1)

```py
def missingNumber(self, nums: List[int]) -> int:
    
    nums.sort()
    last = -1
    for n in nums:
        if last +1 != n:
            return last + 1
        last += 1
        
    return len(nums)
```

## 44. Container with most water

Keep a left and right pointer, and increment the limiting one (smaller one) towards the middle until it isn't the smallest. Keep track of the maximum area calculated. This works because if we are investigating a limiting side and non-limiting side, moving the non-limiting side will never make a greater result, so we should only be moving the limiting side.

- Time O(n)
- Space O(1)

```py
def maxArea(self, height: List[int]) -> int:
    
    l, r = 0, len(height)-1
    
    ret = -1
    
    while l < r:
        
        if height[l] < height[r]:
            ret = max(ret, (r-l)*height[l])
            l += 1
            
        else:
            ret = max(ret, (r-l)*height[r])
            r -= 1        
        
    return ret
```

## 45. Maximum Product Subarray

My solution to this problem came from the observation that the solution will be 
  A) A subarray between either the end points or 0s
  B) 0 itself

For A, if a bounded subarray is positive, it can be proposed as a maximum, but if it is negative then you can extract a positive array by finding the side which contains the largest negative product and dividing. This negative product will be on the boundary on either the left or the right. 

First I tried finding both these values for each bounded subarray, but this was messy. Instead I wrote a helper function to find the negative product of each bounded array on the left and ran it twice with the array and then the reversed array. The maximum of these is the return value. 

The helper function works like so:
- store variables for the current product, the previous product, the best value, the current first negative product and the index that the negative product was found.
- for each number
  - update the current product and previous product
  - if the current product becomes negative, update the first negative product and its corresponding index. Never update again until specified.
  - if the current product becomes 0 or we reach the end
    - if the current product is 0 
      - backtrack the previous product IF there is one
      - update the best value to 0 if it is better (backtracking may find a negative value)
    - if current product is positive or we just found the first negative product
      - update best to current product if it is better
    - else update best to current product / first negative product
    - reset product and first negative variables

- Time O(n^2)?
- Space O(1)

```py

def maxProduct(self, nums: List[int]) -> int:
    return max(self.one_way(nums), self.one_way(nums[::-1]))

def one_way(self, nums):
    
    so_far, prev_so_far = None, None
    initial, initial_index = 1, -1

    best = -float('inf')

    for i, n in enumerate(nums):
        
        so_far, prev_so_far = (so_far or 1)*n, so_far

        if so_far < 0 and initial > 0:
            initial, initial_index = so_far, i

        if n == 0 or i == len(nums)-1:

            if n == 0:
                best = max(best, 0)

                if prev_so_far:
                    i -= 1
                    so_far = prev_so_far

            if so_far > 0 or initial_index == i:
                best = max(best, so_far)
            else:
                best = max(best, so_far/initial)
        
            so_far =  None
            initial, initial_index = 1, -1

    return int(best)
```

## 46. Clone Graph

Keep a map of old to new nodes, and use the keyset as a set of visited noded. When there are no more neighbours to search you can fill in the properties of the node.

- Time O(n)
- Space O(n)

```py
def dfs(self, node, visited):

    visited[node.val] = Node()
    
    for n in node.neighbors:
        if n.val not in visited:
            self.dfs(n, visited)

    visited[node.val].val = node.val
    visited[node.val].neighbors = [visited[n.val] for n in node.neighbors]
            
def cloneGraph(self, node: 'Node') -> 'Node':
    
    if not node:
        return None
    
    visited = dict()
    self.dfs(node, visited)
    return visited[node.val]
```

## 47. Word Search 

Do a dfs on each element, modifying the array temporarily to act as a visited reference. 

- Time O((mn)^2)
- Space O(1)

```py
def dfs(self, r,c,board,word, i):
        
    if len(word) == i:
        return True
    
    if not(0 <= r < len(board) and 0 <= c < len(board[r])):
        return False
    
    if board[r][c] != word[i]:
        return False

    board[r][c] = '.'
    ret = any([self.dfs(x, y, board, word, i+1) for x, y in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]])
            
    board[r][c] = word[i]
    
    return ret
    
def exist(self, board: List[List[str]], word: str) -> bool:
    
    for r in range(len(board)):
        for c in range(len(board[r])):
            if self.dfs(r,c, board, word, 0):
                return True
                
    return False
```
## 48. Word Search II

Do a DFS starting at each index. Use a trie of the dictionary words for searching. When you find a word, delete the EOW symbol from the trie and prune after every search. To track the visisted nodes, convert letters to uppercase temporarily.

- Time O(wn^2)
- Space O(n + w)

```py
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        trie = {}
        for words in words:
            head = trie
            for w in words:    
                if w not in head:
                    head[w] = {}
                head = head[w]
            head['.'] = '.'

        ans = []
        for j in range(len(board)):
            for i in range(len(board[j])):
                if i == 1 and j == 3:
                    print(board, trie)
                ret = self.dfs(board, i, j, '', trie)
                if ret:
                    ans += ret

                    self.prune(trie)

        return ans

    def prune(self, trie):

        if not trie:
            return True

        if trie == '.':
            return False

        keys = [_ for _ in trie.keys()]

        for k in keys:
            if self.prune(trie[k]):
                del trie[k]
        
        return len(trie) == 0
                
    def dfs(self, board, i, j, so_far, trie):

        if not (0 <= i < len(board[0]) and 0 <= j < len(board)):
            return None

        L = board[j][i]

        if L.isupper():
            return None

        if L not in trie:
            return None

        board[j][i] = L.upper()

        words = []
        if '.' in trie[L]:
            del trie[L]['.']
            words.append(so_far + L)
        
        for di, dj in [(1,0), (0,1), (-1,0), (0, -1)]:
            Di, Dj = i + di, j + dj
            ret = self.dfs(board, Di, Dj, so_far+L, trie[L])
            if ret:
                words += ret 

        board[j][i] = L.lower() 

        if len(words) == 0:
            return None
        return words
```
## 49. Merge Intervals

Sort the array by start time and iterate left to right. Whenever a start date is after the current proposed end date, create the new interval and creat a new proposal. If it is conflicting, update the current proposal.

- Time O(nlogn)
- Space O(n)

```py
def merge(self, intervals: List[List[int]]) -> List[List[int]]:

    intervals.sort(key=lambda i: i[0])         
    ans = []

    SE = intervals[0]
    for i in range(1,len(intervals)):

        if intervals[i][0] <= SE[1] <= intervals[i][1]:
            SE[1] = intervals[i][1]

        elif SE[1] < intervals[i][0]:
            ans.append(SE)
            SE = intervals[i]

    ans.append(SE)
    return ans
```

## 50. Subtree of Another Tree

Create a function to check for tree equality. At every step, check for equality then if false recursively run solution on left and right substree.

- Time O(n)
- Space O(1)

```py
def equals(self, a, b):
    
    if not a and not b:
        return True

    if not a or not b:
        return False

    if a.val == b.val:
        return self.equals(a.left, b.left) and self.equals(a.right, b.right)
    
    return False

def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
    
    if not root and subRoot:
        return False

    if self.equals(root, subRoot):
        return True

    return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
```

## 51. Search in Rotated Sorted Array

Do a binary search, but check if there's a discontinuity in the left side. If there isn't then check the target is in the bounds, and if there is then check it could be in the discontinued range. 

- Time O(logn)
- Space O(1)

```py
def search(self, nums: List[int], target: int) -> int:
    i, j = 0, len(nums)-1

    while i <= j:

        mid = (i+j)//2
        if nums[mid] == target:
            return mid
        
        if (nums[i] <= target < nums[mid]) and nums[i] <= nums[mid]:
            j = mid -1
            continue
        
        if (nums[i] <= target or target < nums[mid]) and  nums[i] > nums[mid]:
            j = mid -1
            continue
        
        i = mid + 1

    return -1
```
## 52. Jump Game

Modify the array to be the index plus value. Make two cursors that define a range initially at [0,1), and find the max value. Move the range to have the left cursor at the end of the last range and right cursor to be the max. When your right cursor is at or over the end of the array return True. If the range doesn't move return False.

- Time O(n)
- Space O(1)

```py
def canJump(self, nums: List[int]) -> bool:

    nums = [num+i for i, num in enumerate(nums)]

    i, j = 0, 1

    while j < len(nums):
        furthest = max(nums[i:j])

        if furthest == j-1:
            return False
        
        i = j
        j = furthest + 1

    return True
```

## 53. Find Median From Data Stream

Keep a min heap and max heap either side of the middle two numbers. Keep the heaps balanced with the heap on the right (min heap) never being more than 1 element less than the heap on the left (max heap). When a new number is added, put it on the left if it is lessthan the min of the left heap, otherwise put in on the right. If the right heap grows too big, pop its min and add it to the left heap, and visa-versa. To find the median, take the average if the heaps are the same size, otherwise take the left max (as it is garunteed to not be smaller).

- Time \[addNum\] O(log(n)) \[findMedian\] O(log(n))
- Space O(n)

```py
    def __init__(self):
        # max heap on the left
        self.L = []
        # min heap on the right
        self.R = []

    def addNum(self, num: int) -> None:
        
        if not self.L or num < -self.L[0]:
            heapq.heappush(self.L, -num)
        else:
            heapq.heappush(self.R, num)
        
        if len(self.R) > len(self.L):
            pop = heapq.heappop(self.R)
            heapq.heappush(self.L, -pop)
            return

        if len(self.L) > len(self.R) + 1:
            pop = heapq.heappop(self.L)
            heapq.heappush(self.R, -pop)
            return

    def findMedian(self) -> float:

        l = -self.L[0]

        if self.R:
            r =  self.R[0]
            
            if len(self.L) == len(self.R):
                return (l + r)/2
        
        return l
```
## 54. Group Anagrams

Keep a dictionary of all anagrams, using the sorted tuple as a key. The value is a list of words under that anagram key.

- Time O(n)
- Space O(n)

```py
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    
    groups = dict()

    for s in strs:

        key = tuple(sorted(s))

        groups[key] = groups.get(key, []) + [s]

    return [_ for _ in groups.values()]
```

## 56. Palindromic Substrings

Step through the indexes of the input in steps of 0.5, and get the left and right elements either side and keep adding to the total until they aren't equal.

- Time O(n^2)
- Space O(1)


```py
def countSubstrings(self, s: str) -> int:

    substrings = 0
    for i in range(len(s)*2):

        mid = i/2

        l = int(mid)
        r = round(mid + 0.001)

        while l >= 0 and r < len(s):

            if s[l] != s[r]:
                break
            substrings += 1
            l -= 1
            r += 1

    return substrings
```

## 57. Top K Frequent Elements

Create a dictionary of frequencies and use the frequencies as a key for a max heap. Pull k elements from the heap and return.

- Time O(n)
- Space O(n)

```py
def topKFrequent(self, nums: List[int], k: int) -> List[int]:
    freq = dict()

    for n in nums:
        freq[n] = freq.get(n, 0) + 1

    heap = [(-f, v) for v, f in freq.items()]
    heapq.heapify(heap)

    ret = list()

    for i in range(k):
        ret.append(heapq.heappop(heap)[1])

    return ret
```

Alternatively, you could sort the key value pairs of the frequency map and take the top k elements. This has a complexity of ``O(n logn)``.

## 58. Same Tree

DFS: base case is when both are null (True), one is null (False), or their values are different (False). Else return the AND of sameTree(left) and sameTree(right).

- Time O(n)
- Sapce O(height)

```py
def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    
    if not p and not q:
        return True
    
    if not p or not q:
        return False
    
    if p.val != q.val:
        return False
    
    return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```
## 59. Longest Palindromic Substring

Similar to number of palindromic substrings, you simply find every palindromic substring and update the max value when you find a longer one and save the string through slicing.

- Time O(n^2)
- Space O(1)

```py
def longestPalindrome(self, s: str) -> str:
    longest = 0
    ans = ""
    for i in range(len(s)*2):

        mid = i/2

        l = int(mid)
        r = round(mid + 0.001)

        while l >= 0 and r < len(s):

            if s[l] != s[r]:
                break
            
            
            if r-l+1 > longest:
                ans = s[l:r+1]
                longest = r-l+1
            l -= 1
            r += 1

    return ans
```
## 60. Maximum Depth of Binary Search Tree

Do a dfs and count the depth.

- Time O(n)
- Space O(log(n))

```py 
def dfs(self, root, count):
    if not root:
        return count
    
    L = self.dfs(root.left, count+1)
    R = self.dfs(root.right, count+1)
    
    return max(L, R)

def maxDepth(self, root: Optional[TreeNode]) -> int: 
    return self.dfs(root, 0)
```

## 61. Sum of Two Integers

Call me a hack, I don't care. Python integers are simple at the front, a nightmare at the back. I doubt this question was indended to demonstrate the black magic that is pythonic typing, so I will convert it to a more intutive datatype. This solution is identical to ones you would see for java or C. 

Take the XOR to get the addition of the two numbers and a bitwise shift of the AND to get the carry. Add these together until the carry is 0 and return the sum.

- Time O(n)
- Space O(1)

```py
def getSum(a: int, b: int) -> int:
    
    a = np.int64(a)
    b = np.int64(b)

    while b:
        xor = a ^ b
        car = (a & b) << 1

        a = xor
        b = car

    return a
```
## 62. House Robber

Use dynamic programming. Use i and j to store a window of max values. For a number, n, the new max will be the max of i + n and j. Move the window.

- Time O(n)
- Space O(1)

```py
def rob(self, nums: List[int]) -> int:
    
    i, j = 0, 0

    for n in range(len(nums)):
        tmp = j
        j = max(j, i + nums[n])
        i = tmp

    return j
```

## 63. House Robber II

Simply use solution from House Robber and evaluate twice. Since houses 0 and -1 can't both be robbed, we can find the solutions for all the houses except each one of them and get the max. 

- Time O(n)
- Soace O(1)

```py
def rob(self, nums: List[int]) -> int:
    if len(nums) < 3:
        return max(nums)

    # from 0 to -1
    i, j = 0, 0
    for n in range(len(nums)-1):
        tmp = j
        j = max(j, i + nums[n])
        i = tmp

    A = j
    
    # from 1 to len(nums)
    i, j = 0, 0
    for n in range(1,len(nums)):
        tmp = j
        j = max(j, i + nums[n])
        i = tmp
    
    B = j

    return max(A, B)
```

## 65. Contains Duplicate

Iterate through the string and keep a set of the numbers encountered. If a number is seen twice return True, else return False.

- Time O(n)
- Space O(n)

```py
def containsDuplicate(self, nums: List[int]) -> bool:
    seen = set()

    for n in nums:
        if n in seen:
            return True
        seen.add(n)
    return False
``` 

Alternatively cast the list to a set and compare the lengths of the set and the original list.

