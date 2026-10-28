* 同一数据和抽样次数；调用者设置 command、reps、trials、seed、cores。
* PURPOSE: 在同一进程内比较首次和后续完整命令耗时。
* INPUTS: input.dta 与调用者提供的局部宏。
* OUTPUTS: timings.dta、success.txt。
version 16.0
set more off
set varabbrev off
set processors `cores'
use "input.dta", clear
postfile timing int trial double seconds double diff_x double diff_z ///
    using "timings.dta", replace
forvalues trial = 0/`trials' {
    local current_seed = `seed' + `trial'
    timer clear 88
    timer on 88
    quietly bdiff, group(group) model(`command') reps(`reps') ///
        seed(`current_seed') nodots
    matrix P = r(P)
    timer off 88
    quietly timer list 88
    local seconds = r(t88)
    local x_row = rownumb(P, "x")
    local z_row = rownumb(P, "z")
    post timing (`trial') (`seconds') (P[`x_row', 1]) (P[`z_row', 1])
    display "Trial `trial' finished: `seconds' seconds"
}
postclose timing
file open receipt using "success.txt", write replace
file write receipt "success"
file close receipt
