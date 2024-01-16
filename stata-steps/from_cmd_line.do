// Run script from ZSH with 
// >>> stata -e stpm2_eksempel_severin.do 
// Here, -e automatically exits Stata  
// Note that created alias stata = StataMP 

use "/Users/sela/Desktop/stata-data/example_data.dta", clear
stpm2 sex rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3  if site23 == 4, scale(hazard) bhazard(rate) iterate(50)  df(4) tvc(rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3) dftvc(2)
