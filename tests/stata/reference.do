* 比较分组系数及 suest 协方差；输入和输出均位于调用者的临时目录。
* PURPOSE: 核对分组估计与联合协方差。
* INPUTS: input.dta；调用者设置 command、joint、cluster。
* OUTPUTS: reference.dta、success.txt。
version 16.0
set more off
set varabbrev off
capture set processors 1
* 调用者先设置 command、joint、cluster 三个 local，再使用 include 执行。
use "input.dta", clear
local comma = strpos(`"`command'"', ",")
local head `"`command'"'
local tail ""
if `comma' > 0 {
    local head = substr(`"`command'"', 1, `comma' - 1)
    local tail = substr(`"`command'"', `comma', .)
}
quietly `head' if group == 0 `tail'
estimates store A0
matrix B0 = e(b)
local n0 = e(N)
quietly `head' if group == 1 `tail'
estimates store A1
matrix B1 = e(b)
local n1 = e(N)
if "`joint'" == "yes" {
    if "`cluster'" == "" {
        quietly suest A0 A1
    }
    else {
        quietly suest A0 A1, vce(cluster `cluster')
    }
    matrix V = e(V)
    local fullnames : colfullnames V
}
postfile handle str32 variable double b0 double b1 double diff ///
    double se double p double n0 double n1 ///
    using "reference.dta", replace
foreach variable in x z {
    local c0 = colnumb(B0, "`variable'")
    local c1 = colnumb(B1, "`variable'")
    local b0 = B0[1, `c0']
    local b1 = B1[1, `c1']
    local difference = `b0' - `b1'
    local se = .
    local p = .
    if "`joint'" == "yes" {
        local j = 0
        foreach name of local fullnames {
            local ++j
            if regexm("`name'", "^A0.*:`variable'$") local j0 = `j'
            if regexm("`name'", "^A1.*:`variable'$") local j1 = `j'
        }
        local variance = V[`j0', `j0'] + V[`j1', `j1'] - 2 * V[`j0', `j1']
        local se = sqrt(`variance')
        local p = chi2tail(1, (`difference')^2 / (`variance'))
    }
    post handle ("`variable'") (`b0') (`b1') (`difference') (`se') (`p') (`n0') (`n1')
}
postclose handle
file open receipt using "success.txt", write replace
file write receipt "success"
file close receipt
