use "P:\Registeravdelingen\dta\Eksempelfil_stpm2_severin.dta", clear

//Installer stpm2 hvis ikke allerede gjort 
*ssc install stpm2

//Estimer modell
stpm2 sex rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3  if site23 == 4, scale(hazard) bhazard(rate) iterate(50)  df(4) tvc(rcs4_age1 rcs4_age2 rcs4_age3 rcs4_age4 stage1_2 stage1_3) dftvc(2)
