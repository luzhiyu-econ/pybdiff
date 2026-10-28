* 原版 SSC bdiff 1.04 的结果，作为兼容性回归基准。
* PURPOSE: 记录原版命令的频数与概率。
* INPUTS: input.dta、命令、方法、重复次数、种子。
* OUTPUTS: legacy.dta、success.txt。
version 13
set more off
set varabbrev off
capture set processors 1
args command method reps seed
set seed `seed'
which bdiff
use "input.dta", clear
local option ""
if "`method'" == "bootstrap" local option "bsample"
bdiff, group(group) model(`command') reps(`reps') seed(`seed') nodots `option'
matrix P = r(P)
local variables : rownames P
local k = rowsof(P)
clear
set obs `k'
generate str244 variable = ""
generate double diff = .
generate double frequency = .
generate double p_value = .
forvalues i = 1/`k' {
    local name : word `i' of `variables'
    replace variable = "`name'" in `i'
    replace diff = P[`i', 1] in `i'
    replace frequency = P[`i', 2] in `i'
    replace p_value = P[`i', 3] in `i'
}
save "legacy.dta", replace
file open receipt using "success.txt", write replace
file write receipt "success"
file close receipt
