//
//  Algorithms.swift
//  LeetCodeAlgorithms
//
//  Created by Shinkangsan on 3/5/17.
//  Copyright © 2017 Sheldon. All rights reserved.
//

import UIKit

class Algorithms: NSObject {
    
    //1 Two sum
    /*
     Given nums = [2, 7, 11, 15], target = 9,
     
     Because nums[0] + nums[1] = 2 + 7 = 9,
     return [0, 1].
     */
    class func twoSum(_ nums:[Int],_ target:Int) -> [Int]{
        
        let res = [Int]()
        
        var dict = [Int:Int]()
        
        for i in 0..<nums.count {
            dict[nums[i]] = i
        }
        
        for i in 0..<nums.count {
            if let ind = dict[target-nums[i]], ind != i {
                return [i,ind]
            }
        }
        return res
    }
    
    //6. Zig - Zag print String
    //The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
    //
    //P   A   H   N
    //A P L S I I G
    //Y   I   R
    //And then read line by line: "PAHNAPLSIIGYIR"
    class func convert(_ s: String, _ numRows: Int) -> String {
        guard numRows > 1 else { return s }
        guard !s.isEmpty else { return s }
        var bagPointer = 0
        var ind = 0
        var counter = 0
        var goingDown = true
        var bags = [[String]]()
        for _ in 0..<numRows {
            bags.append([])
        }
        let chars = s.map({String($0)})
        while ind <= chars.count - 1 {
            bags[bagPointer].append(chars[ind])
            ind += 1
            bagPointer = goingDown ? bagPointer + 1 : bagPointer - 1
            counter += 1
            if counter == numRows-1 {
                counter = 0
                goingDown = !goingDown
            }
        }
        return bags.flatMap({$0}).joined()
    }
    
    //12. Integer to Roman
    /*
     Given an integer, convert it to a roman numeral.
     
     Input is guaranteed to be within the range from 1 to 3999.
     */
    class func intToRoman(_ num: Int) -> String {
        let M:[String] = ["", "M", "MM", "MMM"]
        let C:[String] = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        let X:[String] = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        let I:[String] = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10]
    }
    
    //13. Roman to Integer
    /*
     Given an integer, convert it to a roman numeral.
     
     Input is guaranteed to be within the range from 1 to 3999.
     */
    
    //consider two cases
    //for XIX case, should be X + (IX) = 10+1+(10-2), that's why -prev*2
    //for XII case, just X + I + I = 10+1+1
    
    class func romanToInt(_ s: String) -> Int {
        guard !s.isEmpty else { return 0 }
        
        let dict = ["I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000]
        var result = 0
        var prev = 0
        for char in s {
            if let curr = dict[String(char)] {
                if curr > prev {
                    result = result + curr - prev*2
                    prev = curr
                } else {
                    result += curr
                    prev = curr
                }
            }
        }
        return result
    }
    
    //14. common prefix of string array [String]
    class func longestCommonPrefix(_ strs: [String]) -> String {
        if strs.isEmpty {
            return ""
        } else if strs.count == 1 {
            return strs[0]
        } else {
            var commonPrefix = strs[0]
            for i in 1..<strs.count {
                let ary1 = Array(commonPrefix)
                let ary2 = Array(strs[i])
                
                commonPrefix = ""
                innerLoop: for j in 0..<min(ary1.count, ary2.count) {
                    if ary1[j] == ary2[j] {
                        commonPrefix += String(ary1[j])
                    } else {
                        break innerLoop
                    }
                }
            }
            return commonPrefix
        }
    }
    
    //15. three sum. find three numbers that can sum up to 0 from int array
    class func threeSum(_ nums: [Int]) -> [[Int]] {
        
        var resAry = [[Int]]()
        guard nums.count>=3 else { return resAry }
        //step 1: remove duplicate values
        //  a. form a dictionary with item and apprance time
        //  b. iterate dictionary if val>=2 check possible solution
        //                        if val>=3 and key==0 add [0,0,0] solution
        //  c. remove all duplicate values and form the filtered array
        //  d. iterate through the filtered array an chack posiible solution
        
        var dict = [Int:Int]()
        for i in nums {
            if let val = dict[i] {
                dict[i] = val + 1
            } else {
                dict[i] = 1
            }
        }
        let numsSet = NSSet(array:nums)
        
        for (key,val) in dict {
            if val >= 2 && key != 0 {
                if numsSet.contains(0-2*val) {
                    resAry.append([key,key,0-2*key])
                }
            } else if val >= 3 && key == 0 {
                resAry.append([0,0,0])
            }
        }
        
        let filteredAry:[Int] = Array(numsSet) as! [Int]
        
        for i in 0..<filteredAry.count-2 {
            
            for j in i+1..<filteredAry.count-1 {
                
                let val1 = filteredAry[i]
                let val2 = filteredAry[j]
                
                var ary = filteredAry
                ary.remove(at: j)
                ary.remove(at: i)
                
                if NSSet(array:ary).contains(0-val1-val2) {
                    resAry.append([val1,val2,0-val1-val2])
                }
            }
        }
        return resAry
    }
    
    //16. Three Sum - Closest
    /* Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution
     */
    /* For example, given array S = {-1 2 1 -4}, and target = 1.
     The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
     */
    class func threeSumClosest(_ nums: [Int], _ target: Int) -> Int {
        //we are assuming we will have exact one solution, so that we dont need to take care of the cases when nums.count<3
        //so that sNums[0], sNums[1], sNums[2] will not have index error
        
        let sNums = nums.sorted()
        var res = sNums[0]+sNums[1]+sNums[2]
        
        for i in 0..<sNums.count {
            
            var leftInd = i+1, rightInd = sNums.count-1
            
            while leftInd < rightInd {
                
                let sum = sNums[i]+sNums[leftInd]+sNums[rightInd]
                if sum >= target {
                    rightInd -= 1
                } else {
                    leftInd += 1
                }
                
                if abs(sum-target)<abs(res-target) {
                    res = sum
                }
            }
        }
        return res
    }
    
    //17. Letter Combinations of a Phone Number
    /*
     
     1      2       3
     ABC     DEF
     
     4      5       6
     GHI    JKL     MNO
     
     7      8       9
     PQRS   TUV     WXYZ
     
     0
     
     Input:Digit string "23"
     Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
     */
    class func letterCombinations(_ digits: String) -> [String] {
        guard !digits.isEmpty else { return [] }
        
        let map = [[], [], ["a","b","c"], ["d","e","f"], ["g","h","i"], ["j","k","l"], ["m","n","o"], ["p","q","r","s"], ["t","u","v"], ["w","x","y","z"]]
        
        let chars = digits.map{String($0)}
        
        var resAry: [String] = map[Int(chars[0])!]
        
        guard chars.count >= 2 else {
            return resAry
        }
        
        for i in 1..<chars.count {
            
            let tempAry = resAry
            
            for _ in 1..<map[Int(chars[i])!].count {
                resAry += tempAry
            }
            
            var p = 0
            
            for j in 0..<resAry.count {
                
                resAry[j] = resAry[j] + map[Int(chars[i])!][p]
                
                if (j+1) % tempAry.count == 0 {
                    p += 1
                }
            }
        }
        
        return resAry
    }
    
    //18. Four sum
    /*
     Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.
     */
    /*
     For example, given array S = [1, 0, -1, 0, -2, 2], and target = 0.
     
     A solution set is:
     [
     [-1,  0, 0, 1],
     [-2, -1, 1, 2],
     [-2,  0, 0, 2]
     ]
     */
    class func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
        
        var results = [[Int]]()
        let result = [Int]()
        let sNums = nums.sorted()
        
        self.nSum(sNums,target,4,result,&results)
        
        return results
    }
    
    private class func nSum(_ nums:[Int], _ target: Int, _ N: Int, _ result: [Int], _ results: inout [[Int]]) {
        
        guard
            nums.count>=N &&
            N>=2 &&
            target>=nums[0] &&
            target<=nums[nums.count-1]
        else {
            return
        }
        
        if N == 2 {
            var l = 0, r = nums.count-1
            while l<r {
                if (nums[l] + nums[r] == target) {
                    results.append(result + [nums[l],nums[r]] )
                    l += 1
                    r -= 1
                    //remove possible duplicated result for left pointer
                    if l-1>=0 {
                        while l<r && nums[l]==nums[l-1] {
                            l += 1
                        }
                    }
                    //remove possible duplicated result for right pointer
                    if r+1<=nums.count-1 {
                        while l<r && nums[r]==nums[r+1] {
                            r -= 1
                        }
                    }
                } else if nums[l] + nums[r] > target {
                    r -= 1
                } else {
                    l += 1
                }
            }
        } else {
            for i in 0..<nums.count-N+1 {
                //if condition for removing possible duplicated result
                if i==0 || (i>=1 && nums[i] != nums[i-1]) {
                    // recursively call N-1 sum until N = 2
                    let subNums = [Int]() + nums[i+1..<nums.count]
                    self.nSum(subNums,target-nums[i],N-1,result+[nums[i]],&results)
                }
            }
        }
    }
    
    //19. Remove Nth Node From End of List
    /*
     Given a linked list, remove the nth node from the end of list and return its head.
     
     For example,
     
     Given linked list: 1->2->3->4->5, and n = 2.
     
     After removing the second node from the end, the linked list becomes 1->2->3->5.
     */
    //this solution traverse through the list and record the index 
    //after got the target index, it will traverse again and remove it
    //edge cases are the target is the head itself
    
    //another solution is to use to pointers, keep the gap to be N
    //when the fast pointer reaches end, simply remove the slow pointer node
    
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     public var val: Int
     *     public var next: ListNode?
     *     public init(_ val: Int) {
     *         self.val = val
     *         self.next = nil
     *     }
     * }
     */
    
    class func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        
        guard head != nil && n > 0 else{
            return nil
        }
        
        var curNode = head!
        
        var ind = 0
        
        while curNode.next != nil {
            curNode = curNode.next!
            ind += 1
        }
        
        if n > ind+1 {
            return head
        } else if n == ind+1 {
            if let node = head!.next {
                return node
            } else {
                return nil
            }
        } else {
            
            let targetInd = ind-n+1
            
            curNode = head!
            ind = 0
            
            while ind < targetInd-1 {
                curNode = curNode.next!
                ind += 1
            }
            
            if let node = curNode.next!.next {
                curNode.next = node
            } else {
                curNode.next = nil
            }
            return head
        }
    }
    
    //20. Valid Parentheses
    /*
     Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
     
     The brackets must close in the correct order, "([{}])" and "()[]{}" are all valid but "(]" and "([)]" are not.
     
     */
    func isValid(_ s: String) -> Bool {
        
        let ary = s.map{String($0)}
        
        guard ary.count>=2 && ary.count%2==0 else {
            return false
        }
        
        var parenStack = [String]()
        
        for s in ary {
            
            switch s {
                
            case "(","[","{":
                parenStack.append(s)
            case ")":
                if let last = parenStack.last {
                    if last == "(" {
                        _ = parenStack.popLast()!
                    } else {
                        return false
                    }
                } else {
                    return false
                }
            case "]":
                if let last = parenStack.last {
                    if last == "[" {
                        _ = parenStack.popLast()!
                    } else {
                        return false
                    }
                } else {
                    return false
                }
            case "}":
                if let last = parenStack.last {
                    if last == "{" {
                        _ = parenStack.popLast()!
                    } else {
                        return false
                    }
                } else {
                    return false
                }
            default:
                break
            }
        }
        return parenStack.isEmpty
    }
    
    //21. Merge Two Sorted Lists
    //Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
    
    /**
     * Definition for singly-linked list.
     * public class ListNode {
     *     public var val: Int
     *     public var next: ListNode?
     *     public init(_ val: Int) {
     *         self.val = val
     *         self.next = nil
     *     }
     * }
     */
    
    func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        
        if l1 == nil {
            return l2
        }
        
        if l2 == nil {
            return l1
        }
        
        if l1!.val <= l2!.val {
            l1?.next = mergeTwoLists(l1?.next,l2)
            return l1
        } else {
            l2?.next = mergeTwoLists(l1,l2?.next)
            return l2
        }
    }
    
    //22. Generate Parentheses
    /*    Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
     
     For example, given n = 3, a solution set is:
     
     [
     "((()))",
     "(()())",
     "(())()",
     "()(())",
     "()()()"
     ]
     */
    class func generateParenthesis(_ n: Int) -> [String] {
        var list = [String]()
        generateOneByOne("",&list,n,n)
        return list
    }
    private class func generateOneByOne(_ sublist:String,_ list: inout [String],_ left:Int,_ right:Int){
        if left > right {
            return
        }
        if left > 0 {
            generateOneByOne(sublist + "(" , &list, left-1, right);
        }
        if right > 0 {
            generateOneByOne(sublist + ")" , &list, left, right-1);
        }
        if left == 0 && right == 0 {
            list.append(sublist)
        }
    }
    
    //23. Merge k Sorted Lists
    /*
     Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
     */
    class func mergeKLists(_ lists: [ListNode?]) -> ListNode? {
        var tempList = lists
        var cn = 0
        while cn<tempList.count {
            if tempList[cn] == nil {
                tempList.remove(at:cn)
            } else {
                cn += 1
            }
        }
        if cn == 0 {
            return nil
        }
        if cn == 1 {
            return tempList.first!
        }
        var i = 0
        while i<tempList.count {
            tempList[i+1] = merge2List(tempList[i],tempList[i+1])
            i += 1
        }
        return tempList.last!
    }
    
    private class func merge2List(_ l1:ListNode?, _ l2:ListNode?) -> ListNode? {
        if l1 == nil {
            return l2
        }
        if l2 == nil {
            return l1
        }
        
        if l1!.val < l2!.val {
            l1!.next = merge2List(l1!.next,l2)
            return l1
        } else {
            l2!.next = merge2List(l1,l2!.next)
            return l2
        }
    }
    
    //24. Swap Nodes in Pairs
    /*    Given a linked list, swap every two adjacent nodes and return its head.
     
     For example,
     Given 1->2->3->4, you should return the list as 2->1->4->3.
     
     Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed.
     */
    class func swapPairs(_ head: ListNode?) -> ListNode? {
        guard head != nil && head?.next != nil else {
            return head
        }
        
        let newHead = head?.next
        head?.next = swapPairs((head?.next)?.next)
        newHead?.next = head
        return newHead
    }
    
    //29. Divide Two Integers
    /*    Divide two integers without using multiplication, division and mod operator.
     
     If it is overflow, return MAX_INT.
     */
    class func divide(_ dividend: Int, _ divisor: Int) -> Int {
        
        guard divisor != 0 else {
            return Int.max
        }
        
        guard dividend != 0 else {
            return 0
        }
        
        var sign = 1
        
        if dividend < 0 {
            if divisor > 0 {
                sign = -1
            }
        } else {
            if divisor < 0 {
                sign = -1
            }
        }
        
        var dd = abs(dividend)
        let dr = abs(divisor)
        
        //create result var
        var res = 0
        
        //substract using shift operator, will sub temp,2*temp,4*temp.. until it can not sub
        //then reset the temp back to the original value and do the above step again 
        //while loop will stop until the final dd is not >= dr
        while dd >= dr {
            var i = 1, temp = dr
            while dd >= temp {
                dd -= temp
                res += i
                i <<= 1
                temp <<= 1
            }
        }
        return res*sign
    }
    
    //32. Longest Valid Parentheses
    /*
     Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.
     
     For "(()", the longest valid parentheses substring is "()", which has length = 2.
     
     Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4.
     */
    class func longestValidParentheses(_ s: String) -> Int {
        guard s.count != 0 else { return 0 }
        // We will do two scan only record result when "(" & ")" have same amount
        // When ")" has more amount, we will reset result
        // We have to scan both directions to have the correct result for
        // valid result in the case of valid but unfinished pairs like "(()" <- scan from left to right won't work
        
        // scan from left to right
        var currentMax: Int = 0
        var l: Int = 0
        var r: Int = 0
        for c in s {
            if c == "(" {
                l += 1
            } else {
                r += 1
            }
            if r > l {
                l = 0
                r = 0
            } else if r == l {
                currentMax = max(currentMax, r + l)
            }
        }
        // scan from right to left
        l = 0
        r = 0
        for c in s.reversed() {
            if c == ")" {
                l += 1
            } else {
                r += 1
            }
            if r > l {
                l = 0
                r = 0
            } else if r == l {
                currentMax = max(currentMax, r + l)
            }
        }
        return currentMax
    }
    
    //34. Search for a Range
    /*
     Given an array of integers sorted in ascending order, find the starting and ending position of a given target value.
     
     Your algorithm's runtime complexity must be in the order of O(log n).
     
     If the target is not found in the array, return [-1, -1].
     
     For example,
     Given [5, 7, 7, 8, 8, 10] and target value 8,
     return [3, 4].
     */
    class func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        guard !nums.isEmpty else { return [-1, -1] }
        
        var l: Int = -1
        var r: Int = -1
        var low = 0
        var high = nums.count - 1
        var mid = (low + high) / 2
        // find left range
        while (low <= high) {
            if low == high {
                if nums[low] == target {
                    l = low
                }
                break
            }
            if target > nums[mid] { // [1,1,1,8]
                low = mid + 1
            } else if nums[mid] == target {
                l = mid
                high = mid - 1
            } else { // [1,8,8,9,9,9,9]
                high = mid - 1
            }
            mid = (low + high) / 2
        }
        //find right range
        low = 0
        high = nums.count - 1
        mid = (low + high) / 2
        while (low <= high) {
            if low == high {
                if nums[low] == target {
                    r = low
                }
                break
            }
            if target > nums[mid] { // [1,1,1,8]
                low = mid + 1
            } else if nums[mid] == target {
                r = mid
                low = mid + 1
            } else { // [1,8,8,9,9,9,9]
                high = mid - 1
            }
            mid = (low + high) / 2
        }
        return [l, r]
    }
    
    //35. Search Insert Position
    /*
     Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
     
     You may assume no duplicates in the array.
     
     Example 1:
     
     Input: [1,3,5,6], 5
     Output: 2
     Example 2:
     
     Input: [1,3,5,6], 2
     Output: 1
     Example 3:
     
     Input: [1,3,5,6], 7
     Output: 4
     Example 1:
     
     Input: [1,3,5,6], 0
     Output: 0
     */
    class func searchInsert(_ nums: [Int], _ target: Int) -> Int {
        guard !nums.isEmpty else { return 0 }
        var l = 0
        var r = nums.count - 1
        var mid = (l + r) / 2
        var pre = nums.count
        while (l <= r) {
            if (l == r) {
                if target <= nums[l] {
                    return l
                }
                break
            }
            if target == nums[mid] {
                return mid
            } else if target < nums[mid] {
                pre = mid
                r = mid - 1
            } else {
                l = mid + 1
            }
            mid = (l + r) / 2
        }
        return pre
    }
    
    //39. Combination Sum
    /*
     Given a set of candidate numbers (C) (without duplicates) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
     
     The same repeated number may be chosen from C unlimited number of times.
     
     Note:
     All numbers (including target) will be positive integers.
     The solution set must not contain duplicate combinations.
     For example, given candidate set [2, 3, 6, 7] and target 7,
     A solution set is:
     [
     [7],
     [2, 2, 3]
     ]
     */
    class func combinationSum(_ candidates: [Int], _ target: Int) -> [[Int]] {
        var result = [[Int]]()
        findResult(candidates.sorted(), target, &result, [Int](), 0)
        return result
    }
    private class func findResult(_ candidates: [Int], _ target: Int, _ result: inout [[Int]], _ combination: [Int], _ start: Int) {
        if target == 0 {
            result.append(combination)
        } else if target < 0 {
            return
        } else {
            for i in start..<candidates.count {
                if candidates[i] <= target {
                    var c = combination
                    c.append(candidates[i])
                    findResult(candidates, target - candidates[i], &result, c, i)
                }
            }
        }
    }
    
    //40. Combination Sum II
    /* Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
     
     Each number in C may only be used once in the combination.
     
     Note:
     All numbers (including target) will be positive integers.
     The solution set must not contain duplicate combinations.
     For example, given candidate set [10, 1, 2, 7, 6, 1, 5] and target 8,
     A solution set is:
     [
     [1, 7],
     [1, 2, 5],
     [2, 6],
     [1, 1, 6]
     ]
     */
    class func combinationSum2(_ candidates: [Int], _ target: Int) -> [[Int]] {
        guard !candidates.isEmpty else { return [[Int]]() }
        var result = [[Int]]()
        updateResult(candidates.sorted(), target, &result, [Int](), 0)
        return result
    }
    private class func updateResult(_ candidates: [Int], _ target: Int, _ result: inout [[Int]], _ combination: [Int], _ ind: Int) {
        if target == 0 {
            result.append(combination)
        } else if target < 0 {
            return
        } else {
            for i in ind..<candidates.count {
                if (i > ind && candidates[i] == candidates[i-1]) { continue }
                updateResult(candidates, target - candidates[i], &result, combination + [candidates[i]], i+1)
            }
        }
    }
    
    //41. First Missing Positive
    /*
     Given an unsorted integer array, find the first missing positive integer.
     
     For example,
     Given [1,2,0] return 3,
     and [3,4,-1,1] return 2.
     
     Your algorithm should run in O(n) time and uses constant space. //My solution is nLg(n)
     */
    class func firstMissingPositive(_ nums: [Int]) -> Int {
        let sNums = nums.filter({$0>0}).sorted()
        guard !sNums.isEmpty else { return 1 }
        
        var cur = 0
        for i in 0..<sNums.count {
            if sNums[i] > cur + 1 { return cur + 1 }
            if sNums[i] == cur + 1 {
                cur += 1
            }
        }
        return cur + 1
    }
    
    //42. Trapping Rain Water
    /*
     Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
     
     For example,
     Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
     See image: https://leetcode.com/static/images/problemset/rainwatertrap.png
     See question: https://leetcode.com/problems/trapping-rain-water/description/
     */
    class func trap(_ height: [Int]) -> Int {
        // scan from left to right to record pipes that taller than previous recorded pipe
        var indArray = [Int]()
        var max1 = -1
        for i in 0..<height.count {
            if height[i] > 0 && height[i] >= max1 {
                indArray.append(i)
                max1 = height[i]
            }
        }
        // scan from right to left to record pipes that taller than previous recorded pipe until max height
        let rHeight: [Int] = height.reversed()
        var indArray2 = [Int]()
        var max2 = -1
        for i in 0..<height.count {
            if rHeight[i] == max1 { break }
            if rHeight[i] > 0 && rHeight[i] >= max2 {
                indArray2.append(height.count - 1 - i)
                max2 = rHeight[i]
            }
        }
        // generate useful pipes array
        indArray2 = indArray2.reversed()
        indArray = indArray + indArray2
        guard indArray.count >= 2 else { return  0 }
        var res = 0
        
        // calculate the water in between useful pipes, also deduct the pipes in between useful pipes
        for i in 0..<indArray.count {
            if i > 0 {
                res += min(height[indArray[i]], height[indArray[i-1]]) * (indArray[i] - indArray[i-1] - 1)
                if indArray[i] - indArray[i-1] > 1 {
                    for j in (indArray[i-1]+1)..<indArray[i] {
                        res -= height[j]
                    }
                }
            }
        }
        return res
    }
    
    //49. Group Anagrams
    /*
     Given an array of strings, group anagrams together.
     
     For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"],
     Return:
     
     [
     ["ate", "eat","tea"],
     ["nat","tan"],
     ["bat"]
     ]
     */
    class func groupAnagrams(_ strs: [String]) -> [[String]] {
        guard strs.count>0 else {
            return [[]]
        }
        
        let prime:[String:Int] = ["a":2, "b":3, "c":5, "d":7, "e":11, "f":13, "g":17, "h":19, "i":23, "j":29, "k":31, "l":41, "m":43, "n":47, "o":53, "p":59, "q":61, "r":67, "s":71, "t":73, "u":79, "v":83, "w":89, "x":97, "y":101, "z":103]
        var muls = [Int]()
        
        for s in strs {
            if s.isEmpty {
                muls.append(0)
            } else {
                var mul = 1
                for ch in s {
                    mul = mul * (prime[String(ch)]!)
                }
                muls.append(mul)
            }
        }
        
        var map = [Int:[String]]()
        
        for i in 0..<muls.count {
            if map[muls[i]] != nil{
                var aryCopy:[String] = map[muls[i]]!
                aryCopy.append(strs[i])
                map[muls[i]] = aryCopy
            } else {
                map[muls[i]] = [strs[i]]
            }
        }
        
        var res = [[String]]()
        
        for (_, vals) in map {
            res.append(vals)
        }
        
        return res
    }
    
    //62. Unique Paths
    /*
     A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
     
     The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
     
     ---------------
     |R| | | | | | |
     ---------------
     | | | | | | | |
     ---------------
     | | | | | | |F|
     ---------------
     
     R = Robot, F = Finish
     */
    class func uniquePaths(_ m: Int, _ n: Int) -> Int {
        let tot = m+n-2
        let down = m-1
        //choose m-1 is the amount to go down
        //res = choose m-1 out of m+n-2 = tot!/((m-1)!*(n-1)!)
        if m==0 || n==0 {
            return 0
        }
        if m==1 || n==1 {
            return 1
        }
        return factor(tot)/(factor(down)*factor(tot-down)) //here can be improved with multiplication instead of factorial
    }
    
    class private func factor(_ v:Int) -> Int{
        guard v>0 else {
            return 0
        }
        var res = 1
        var i = v
        while i > 0 {
            res *= i
            i -= 1
        }
        return res
    }
    
    //75. Sort Colors
    /*
     Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.
     
     Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
     
     Note:
     You are not suppose to use the library's sort function for this problem.
     */
    class func sortColors(_ nums: inout [Int]) {
        var l = 0, r = nums.count-1
        var i = 0
        
        while i <= r {
            while (nums[i] == 2 && i < r) {
                nums.swapAt(i, r)
                r -= 1
            }
            while (nums[i] == 0 && i > l) {
                nums.swapAt(i, l)
                l += 1
            }
            i += 1
        }
    }
    
    //91. Decode String
    /*
     A message containing letters from A-Z is being encoded to numbers using the following mapping:
     
     'A' -> 1
     'B' -> 2
     ...
     'Z' -> 26
     Given an encoded message containing digits, determine the total number of ways to decode it.
     
     For example,
     Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
     
     The number of ways decoding "12" is 2.
     */
    class func numDecodings(_ s: String) -> Int {
        
        if s.isEmpty {
            return 0
        }
        
        //put a placehodler 1 at first, which indicates "" has only one way to decode
        //although given empty string we will return 0
        let n = s.count
        var dp = [Int]()
        dp.append(1)
        
        //append 0s for the result n count to make the dp array length: n+1
        for _ in 0..<n {
            dp.append(0)
        }
        
        //check the first character, if 0, dp[1]=0, since 0 has to be part of 10 or 20, else dp[1]=1
        let chars = s.map{String($0)}
        dp[1] = (chars[0] == "0") ? 0 : 1
        
        //return here for the case of "0"
        if s.count == 1 {
            return dp[1]
        }
        
        //accumulate result with previously saved numbers
        for i in 2...n {
            let first = Int(chars[i-1])!
            let second = Int(chars[i-2]+chars[i-1])!
            if(first >= 1 && first <= 9) {
                dp[i] += dp[i-1]
            }
            if(second >= 10 && second <= 26) {
                dp[i] += dp[i-2]
            }
        }
        return dp[n]
    }
    
    //98. Valid Binary Search Tree
    class func isValidBST(_ root: TreeNode?) -> Bool {
        
        return isValidWithin(root, Int.min, Int.max)
    }
    
    class private func isValidWithin(_ root: TreeNode?, _ min:Int, _ max:Int) -> Bool {
        if root == nil {
            return true
        }
        
        if let v = root?.val, v >= max || v <= min {
            return false
        }
        
        return isValidWithin(root?.left, min, (root?.val)!) && isValidWithin(root?.right, (root?.val)!, max)
    }
    
    //Select K nearest points
    //find k nearest points in and points array regarding the target point
    class func kNearest(_ target:CGPoint, _ points:[CGPoint], _ k: Int) -> [CGPoint] {
        
        var heap = [CGPoint]()
        
        guard k>0 else {
            return heap
        }
        
        guard points.count>0 else {
            return heap
        }
        
        for point in points {
            adjustHeap(&heap, newPoint: point, k, target)
        }
        
        return heap
    }
    
    class private func adjustHeap(_ heap: inout [CGPoint], newPoint:CGPoint, _ k: Int, _ targetPoint: CGPoint) {
        
        if heap.isEmpty {
            heap.append(newPoint)
        } else {
            var i = heap.count-1
            wLoop: while i >= 0 {
                if distanceBtwPoints(newPoint, targetPoint) < distanceBtwPoints(heap[i], targetPoint) {
                    i -= 1
                } else {
                    break wLoop
                }
            }
            
            heap.insert(newPoint, at: i+1)
            if heap.count > k {
                heap.removeLast()
            }
        }
    }
    
    class func distanceBtwPoints(_ p1:CGPoint, _ p2:CGPoint) -> Double {
        return pow(Double(p1.x)-Double(p2.x),2) + pow(Double(p1.y)-Double(p2.y), 2)
    }
    
    //114. Flatten Binary Tree to Linked List
    /**
     * Definition for a binary tree node.
     * public class TreeNode {
     *     public var val: Int
     *     public var left: TreeNode?
     *     public var right: TreeNode?
     *     public init(_ val: Int) {
     *         self.val = val
     *         self.left = nil
     *         self.right = nil
     *     }
     * }
     */
    /*
     For example,
     Given
     
         1
        / \
       2   5
      / \   \
     3   4   6
     
     The flattened tree should look like:
     1
      \
      2
       \
       3
        \
        4
         \
         5
          \
          6
     */
    class func flatten(_ root: TreeNode?) {
        if root == nil {
            return
        }
        
        var head = root
        flatten(head?.left)
        flatten(head?.right)
        
        let tempR = head?.right
        
        head?.right = head?.left
        head?.left = nil
        
        while head?.right != nil {
            head = head?.right
        }
        
        head?.right = tempR
        
    }
    
    //125. Valid Palindrome
    /*
     Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
     
     For example,
     "A man, a plan, a canal: Panama" is a palindrome.
     "race a car" is not a palindrome.
     
     Note:
     Have you consider that the string might be empty? This is a good question to ask during an interview.
     
     For the purpose of this problem, we define empty string as valid palindrome.
     
     */
    class func isPalindrome(_ s: String) -> Bool {
        if s.isEmpty {
            return true
        }
        var pdStr = s
        let charSet = NSCharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        pdStr = pdStr.components(separatedBy: charSet.inverted).joined().lowercased()
        
        if pdStr.isEmpty || pdStr.count == 1 {
            return true
        }
        
        let charAry = pdStr.map{String($0)}
        for i in 0...(charAry.count-1)/2 {
            if charAry[i] != charAry[charAry.count-1-i] {
                return false
            }
        }
        return true
    }
    
    //198. House Robber
    /*
     You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
     
     Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
     */
    class func rob(_ nums: [Int]) -> Int {
        
        if nums.isEmpty {
            return 0
        }
        
        if nums.count == 1 {
            return nums[0]
        }
        
        if nums.count == 2 {
            return max(nums[0],nums[1])
        }
        
        var dp = [Int]()
        dp.append(nums[0])
        dp.append(max(nums[0],nums[1]))
        for _ in 2...nums.count-1 {
            dp.append(0)
        }
        
        for i in 2...nums.count-1 {
            dp[i] = max(nums[i] + dp[i-2], dp[i-1])
        }
        
        return dp.last!
    }
    
    //209. Minimum Size Subarray Sum
    /*
     Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.
     
     For example, given the array [2,3,1,2,4,3] and s = 7,
     the subarray [4,3] ≥ 7 has the minimal length under the problem constraint.
     */
    
    //step to solve this in one pass, first move the right side towards right
    //check if sum >= s, if not it means alfter search all it failed to get the result
    //next move left side towards right
    //check if sum < s, if not it means no result, no update length
    
    //one thing to be noticed that "r" and "i" are one step faster than the wanted index
    
    class func minSubArrayLen(_ s: Int, _ nums: [Int]) -> Int {
        
        var l = 0, r = 0
        var sum = 0
        var len = Int.max
        
        while r < nums.count {
            while sum < s && r < nums.count {
                sum += nums[r]
                r += 1
            }
            
            if sum >= s {
                len = min(len,(r-1)-l+1)
            } else {
                break
            }
            
            while sum >= s && l < r {
                sum -= nums[l]
                l += 1
            }
            
            if sum < s {
                len = min(len,(r-1)-(l-1)+1)
            }
        }
        
        if len == Int.max {
            len = 0
        }
        return len
    }
    
    //221. Maximal Square
    /**
     Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
     
     For example, given the following matrix:
     
     1 0 1 0 0
     1 0 1 1 1
     1 1 1 1 1
     1 0 0 1 0
     Return 4.
     
     I was using brute force solution, see DP solution here: https://leetcode.com/problems/maximal-square/solution/
     */
    class func maximalSquare(_ matrix: [[Character]]) -> Int {
        guard !matrix.isEmpty else { return 0 }
        guard !matrix.first!.isEmpty else { return 0 }
        
        var currentLength: Int = 0
        for i in 0..<matrix.count {
            for j in 0..<matrix.first!.count {
                
                var length: Int = 0
                if matrix[i][j] == "0" {
                    continue
                } else {
                    length = 1
                    spreadSquare(i, j, &length, matrix)
                }
                
                currentLength = max(currentLength, length)
            }
        }
        return currentLength * currentLength
    }
    
    class private func spreadSquare(_ x: Int, _ y: Int, _ length: inout Int, _ matrix: [[Character]]) {
        guard x+length < matrix.count && y+length < matrix.first!.count else { return }
        for i in x...(x+length) {
            if matrix[i][y+length] == Character("0") {
                return
            }
        }
        for i in y...(y+length) {
            if matrix[x+length][i] == Character("0") {
                return
            }
        }
        length += 1
        spreadSquare(x, y, &length, matrix)
    }
    
    //239. Sliding Window Maximum
    /*
     Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.
     
     For example,
     Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.
     
     Window position                Max
     ---------------               -----
     [1  3  -1] -3  5  3  6  7       3
     1 [3  -1  -3] 5  3  6  7       3
     1  3 [-1  -3  5] 3  6  7       5
     1  3  -1 [-3  5  3] 6  7       5
     1  3  -1  -3 [5  3  6] 7       6
     1  3  -1  -3  5 [3  6  7]      7
     Therefore, return the max sliding window as [3,3,5,5,6,7].
     
     Note:
     You may assume k is always valid, ie: 1 ≤ k ≤ input array's size for non-empty array.
     */
    class func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
        
        guard !nums.isEmpty && k <= nums.count else {
            return []
        }
        //O(n*k)
        var tempMax = Int.min
        var tempMaxInd = -1
        var res = [Int]()
        
        for i in 0...nums.count-k {
            if tempMaxInd>=i && tempMaxInd<=i+k-1 {
                tempMax = max(tempMax,nums[i+k-1])
                res.append(tempMax)
            } else {
                tempMax = Int.min
                for j in i..<i+k {
                    tempMax = max(tempMax,nums[j])
                }
                res.append(tempMax)
            }
        }
        return res
    }
    
    //350. Intersection of Two Arrays II
    /*
     Given two arrays, write a function to compute their intersection.
     
     Example:
     Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].
     
     Note:
     Each element in the result should appear as many times as it shows in both arrays.
     The result can be in any order.
     */
    //sort and then compare
    class func intersect(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        
        guard !nums1.isEmpty && !nums2.isEmpty else {
            return []
        }
        
        let sNums1 = nums1.sorted()
        let sNums2 = nums2.sorted()
        
        var inter = [Int]()
        
        var i = 0, j = 0
        
        while i<nums1.count && j<nums2.count {
            
            if sNums1[i] < sNums2[j] {
                i += 1
            } else if sNums1[i] == sNums2[j] {
                inter.append(sNums1[i])
                i += 1
                j += 1
            } else {
                j += 1
            }
        }
        
        return inter
        
    }
    
    //367. Valid Perfect Square
    /**
     Given a positive integer num, write a function which returns True if num is a perfect square else False.
     
     Note: Do not use any built-in library function such as sqrt.
     
     Example 1:
     
     Input: 16
     Returns: True
     Example 2:
     
     Input: 14
     Returns: False
     */
    class func isPerfectSquare(_ num: Int) -> Bool {
        for i in 1...num {
            let x: Double = Double(num / i)
            let y: Double = Double(i)
            if y > x {
                return false
            }
            if num % i == 0 && x == Double(i) {
                return true
            }
        }
        return false
    }
    
    // First unique char in a string
    class func findFirstUniqueChar(chars: [String]) -> String {
        guard !chars.isEmpty else { return "" }
        var charsSet = Set<String>()
        var repeatedCharsSet = Set<String>()
        for char in chars {
            let (res, _) = charsSet.insert(char)
            if res == false {
                _ = repeatedCharsSet.insert(char)
            }
        }
        for i in 0..<chars.count {
            if !repeatedCharsSet.contains(chars[i]) {
                return chars[i]
            }
        }
        return ""
    }
    
    //414. Third Maximum Number
    //Given a NON-EMPTY array of integers, return the third maximum number in this array. If it does not exist, return the maximum number. The time complexity must be in O(n).
    /**
     Example 1:
     Input: [3, 2, 1]
     
     Output: 1
     
     Explanation: The third maximum is 1.
     Example 2:
     Input: [1, 2]
     
     Output: 2
     
     Explanation: The third maximum does not exist, so the maximum (2) is returned instead.
     Example 3:
     Input: [2, 2, 3, 1]
     
     Output: 1
     
     Explanation: Note that the third maximum here means the third maximum distinct number.
     Both numbers with value 2 are both considered as second maximum.
     */
    class func thirdMax(_ nums: [Int]) -> Int {
        var top1: Int?
        var top2: Int?
        var top3: Int?
        
        for i in 0..<nums.count {
            if nums[i] == top1 || nums[i] == top2 || nums[i] == top2 {
                continue
            } else if top1 == nil {
                top1 = nums[i]
            } else if top1! < nums[i] {
                top3 = top2
                top2 = top1
                top1 = nums[i]
            } else if top2 == nil, nums[i] < top1! {
                top2 = nums[i]
            } else if top2 != nil, nums[i] > top2! {
                top3 = top2
                top2 = nums[i]
            } else if top3 == nil, top2 != nil, nums[i] < top2! {
                top3 = nums[i]
            } else if top3 != nil, top2 != nil, nums[i] < top2!, nums[i] > top3! {
                top3 = nums[i]
            }
        }
        return top3 == nil ? top1! : top3!
    }
    
    //433. Minimum Genetic Mutation
    /**
     A gene string can be represented by an 8-character long string, with choices from "A", "C", "G", "T".
     
     Suppose we need to investigate about a mutation (mutation from "start" to "end"), where ONE mutation is defined as ONE single character changed in the gene string.
     
     For example, "AACCGGTT" -> "AACCGGTA" is 1 mutation.
     
     Also, there is a given gene "bank", which records all the valid gene mutations. A gene must be in the bank to make it a valid gene string.
     
     Now, given 3 things - start, end, bank, your task is to determine what is the minimum number of mutations needed to mutate from "start" to "end". If there is no such a mutation, return -1.
     
     Note:
     
     Starting point is assumed to be valid, so it might not be included in the bank.
     If multiple mutations are needed, all mutations during in the sequence must be valid.
     You may assume start and end string is not the same.
     Example 1:
     
     start: "AACCGGTT"
     end:   "AACCGGTA"
     bank: ["AACCGGTA"]
     
     return: 1
     Example 2:
     
     start: "AACCGGTT"
     end:   "AAACGGTA"
     bank: ["AACCGGTA", "AACCGCTA", "AAACGGTA"]
     
     return: 2
     Example 3:
     
     start: "AAAAACCC"
     end:   "AACCCCCC"
     bank: ["AAAACCCC", "AAACCCCC", "AACCCCCC"]
     
     return: 3
     */
    class func minMutation(_ start: String, _ end: String, _ bank: [String]) -> Int {
        guard start.count == end.count, start.count == 8 else { return -1 }
        
        var mutableBank: [String] = bank
        var searchCount: Int = 0
        
        var reversedMutableBank: [String] = bank.reversed()
        var reversedSearchCount: Int = 0
        
        let res1 = searchAndRemove(start, end, &mutableBank, &searchCount)
        let res2 = searchAndRemove(start, end, &reversedMutableBank, &reversedSearchCount)
        
        if !res1 && !res2 {
            return -1
        } else if res1 && res2{
            return min(searchCount, reversedSearchCount)
        } else {
            return res1 ? searchCount: reversedSearchCount
        }
        
    }
    
    private class func searchAndRemove(_ start: String, _ end: String, _ bank: inout [String], _ count: inout Int) -> Bool {
        if start == end { return true }
        let tempBank = bank
        let tempCount = count
        for gene in bank {
            bank = tempBank
            count = tempCount
            if isOneMutationAway(start, gene) {
                let ind = (bank.index(of: gene))!
                bank.remove(at: ind)
                count += 1
                if searchAndRemove(gene, end, &bank, &count) {
                    return true
                } else {
                    continue
                }
            } else {
                continue
            }
        }
        return false
    }
    
    private class func isOneMutationAway(_ l: String, _ r: String) -> Bool {
        guard l.count == r.count else { return false }
        var j = 0
        for i in 0..<l.count {
            if Array(l)[i] != Array(r)[i] {
                j += 1
            }
        }
        return j == 1
    }
    
    //721. Accounts Merge
    /**
     Given a list accounts, each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.
     
     Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some email that is common to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.
     
     After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.
     
     Example 1:
     Input:
     accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
     Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
     Explanation:
     The first and third John's are the same person as they have the common email "johnsmith@mail.com".
     The second John and Mary are different people as none of their email addresses are used by other accounts.
     We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'],
     ['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
     Note:
     
     The length of accounts will be in the range [1, 1000].
     The length of accounts[i] will be in the range [1, 10].
     The length of accounts[i][j] will be in the range [1, 30].
     */
    
    static var graph = [String: [String]]()
    static var visited = Set<String>()
    
    class func accountsMerge(_ accounts: [[String]]) -> [[String]] {
        guard !accounts.isEmpty else { return [] }
        var names = [String: String]()
        for account in accounts {
            for email in Array(account[1...]) {
                if let v = graph[account[1]] {
                    if !v.contains(email) {
                        graph[account[1]] = v + [email]
                    }
                } else {
                    graph[account[1]] = [email]
                }
                if let v = graph[email] {
                    if !v.contains(account[1]) {
                        graph[email] = v + [account[1]]
                    }
                } else {
                    graph[email] = [account[1]]
                }
                names[email] = account[0]
            }
        }
        
        var groups = [[String]]()
        for account in accounts {
            if !visited.contains(account[1]) {
                let group = addToList(account[1], [String]())
                groups.append(group)
            }
        }
        
        var res = [[String]]()
        for group in groups {
            guard let name = names[group[0]] else { continue }
            res.append([name]+group.sorted())
        }
        return res
    }
    
    private class func addToList(_ email: String, _ list: [String]) -> [String] {
        if visited.contains(email) { return list }
        guard let emails = graph[email], !emails.isEmpty else { return list }
        visited.insert(email)
        var res = list + [email]
        for item in emails {
            res += addToList(item, list)
        }
        return res
    }
}
