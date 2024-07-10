'''
Tests 1 to 4 are general logic-tests, 5 and onwards are specific logic-tests for the rules
5: A must be at exactly right of B
6: A must be next to B : loc(A)=loc(B)+1 or loc(B)-1
7: boundary conditions
 - each plant can only be planted at one position
 - one position can only hold one plant
8: A or B must be at position x
9: A must be at exactly left of B
10: A cannot be next to B
11: If A is in position x, then B must be in position y
'''

from rules import NextToConstraint, NotNextToConstraint, ToTheRightConstraint, ToTheLeftConstraint, IfThenNotNextToConstraint, \
    EitherOrConstraint, IfThenConstraint


layout1 = ['*','*','Peas','*','*']
layout2 = ['Tomato','*','*','*','*']
layout3 = ['Carrot','*','*','*','*','*']
layout4 = ['Corn','*','*','*','*','*']
layout5 = ['*','Tomato','*','*']
layout6 = ['*','Peas','*','*']
layout7 = ['Berries','Tomato','Peas','*']
layout8 = ['*','Tomato','*','*']
layout9 = ['Berries', '*', 'Peas', '*']
layout10 = ['*','Peas','*','*']
layout11 = ['*','Carrot','Tomato','*']


rules1 = [NextToConstraint("Peas","Carrot"), ToTheRightConstraint("Grape","Carrot"), IfThenNotNextToConstraint("Peas","Carrot","Berries")]
rules2 = [NextToConstraint("Tomato","Carrot"), ToTheRightConstraint("Grape","Carrot"), ToTheRightConstraint("Peas","Grape"),
          IfThenNotNextToConstraint("Grape","Carrot","Berries")]
rules3 = [ToTheRightConstraint("Berries","Corn"), ToTheRightConstraint("Corn","Grape"), ToTheRightConstraint("Tomato","Berries"),
          EitherOrConstraint("Peas","Grape", 3), EitherOrConstraint("Peas","Carrot", 2)]
rules4 = [NextToConstraint("Berries", "Grape"), ToTheRightConstraint("Tomato", "Carrot"), NextToConstraint("Peas", "Carrot"),
          NextToConstraint("Tomato", "Berries")]
rules5 = [EitherOrConstraint("Peas", "Carrot", 1), ToTheRightConstraint("Peas", "Tomato")]
rules6 = [EitherOrConstraint("Peas", "Tomato", 1), NextToConstraint("Carrot", "Peas")]
rules7 = []
rules8 = [EitherOrConstraint("Peas", "Tomato", 1), EitherOrConstraint("Carrot", "Tomato", 3)]
rules9 = [ToTheLeftConstraint("Tomato", "Peas"), ToTheLeftConstraint("Peas", "Carrot")]
rules10 = [NotNextToConstraint("Corn", "Peas"), NotNextToConstraint("Berries", "Corn")]
rules11 = [IfThenConstraint("Tomato", 3, "Berries", 1)]



to_be_assigned1 = ['Carrot','Tomato','Grape','Berries']
to_be_assigned2 = ['Carrot','Peas','Grape','Berries']
to_be_assigned3 = ['Peas','Grape','Corn','Berries','Tomato']
to_be_assigned4 = ['Grape','Carrot', "Berries", "Peas", 'Tomato']
to_be_assigned5 = ['Berries','Peas','Carrot']
to_be_assigned6 = ['Berries','Tomato','Carrot']
to_be_assigned7 = ['Carrot']
to_be_assigned8 = ['Berries','Peas','Carrot']
to_be_assigned9 = ["Tomato", "Carrot"]
to_be_assigned10 = ["Corn", "Berries", "Grape"]
to_be_assigned11 = ["Berries", "Grape"]


solution1 = "C4&T2&G5&B1"
solution2 = "C2&P4&G3&B5"
solution3 = "P2&G3&Cn4&B5&T6"
solution4 = "G6&C3&B5&P2&T4"
solution5 = "B4&P3&C1"
solution6 = "B4&T1&C3"
solution7 = "C4"
solution8 = "B4&P1&C3"
solution9 = "C4&T2&P3&B1"
solution10 = "P2&G3&B1&Cn4"
solution11 = "B1&C2&T3&G4"
