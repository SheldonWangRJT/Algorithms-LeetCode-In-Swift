//
//  ViewController.swift
//  LeetCodeAlgorithms
//
//  Created by Shinkangsan on 3/5/17.
//  Copyright © 2017 Sheldon. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        
        //q1
        let a1 = Algorithms.twoSum([1,2,3,4], 3)
        print(a1)
        
        
        Algorithms.threeSum([-1,0,1,2,-1,-4])
        Algorithms.fourSum([-2,-1,0,0,1,2], 0)
        
        let list = Algorithms.generateParenthesis(2)
        print(list)
    }


}

