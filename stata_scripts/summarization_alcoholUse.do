
use "E:\SOEP_DATA\Stata\pl.dta" ,clear

keep pid syear ple0090 ple0091 ple0092 ple0093

keep if inlist(syear,2006,2008,2010)

mvdecode _all, mv(-5 -3 -2 -1)

* Alcohol consumtion
recode ple0090 (1=3) (2=2) (3=1) (4=0)
recode ple0091 (1=3) (2=2) (3=1) (4=0)
recode ple0092 (1=3) (2=2) (3=1) (4=0)
recode ple0093 (1=3) (2=2) (3=1) (4=0)

alpha ple0090 ple0091 ple0092 ple0093, d i g(alcohol_combined)
alpha ple0090 ple0091, d i g(alcohol_16)
alpha ple0092 ple0093, d i g(alcohol_hard)

rename ple0092 spirits
rename ple0093 mixed

label variable alcohol_combined Alcoholall4
label variable alcohol_16 Alcohol_from_16_years
label variable alcohol_hard Alcohol_from_18_years


keep pid syear alcohol_combined alcohol_16 alcohol_hard spirits mixed

saveold "C:\Users\rapha\Documents\PanelDataProject\alcoholScale",replace


