using Plots
using Colors
using Random
using XLSX
using Optim
using Dates
using DayCounts
using Statistics
using DataFrames
using CSV

# Cycle corporate colours
cgold = colorant"rgb(136,106,38)"
cgreen = colorant"rgb(215,237,199)"
crosarot = colorant"rgb(237,196,196)"

# Some Randomness
Random.seed!(42)

#cd()
#cd("/Users/tc/OneDrive - Conring Frisius/Dan")

# Cycle inputs via excel sheet for constructed portfolio
fund = convert(Array{Int,2},replace(XLSX.readdata("Cycle_inputs.xlsx","Input!C4:AT18"), missing=>0))
prob = convert(Array{Float32,2},replace(XLSX.readdata("Cycle_inputs.xlsx","Input!C30:AT103"), missing=>0))
prob = prob[setdiff(1:end, [5:5:74...]), :]
dist = convert(Array{Float32,2},replace(XLSX.readdata("Cycle_inputs.xlsx","Input!C109:AT123"), missing=>0))

# function to compute the action at a milestone
# Input: the probalities
# Output: Integer indicating the action: 0 stay in phase and improve, 1 success, 2 success without topup, 3 failure
@inline function phase_transition_check( success, successwotopu, stall, failure)
   r = rand()
   if r <= success
      # success
     return 1
  elseif r > success && r <= ( success + successwotopu )
     # success without topup
     return 2
   elseif r > ( success + successwotopu ) && r <= ( success + successwotopu + stall )
      # stay in phase and improve
      return 0
   else
      # failure
      return 3
   end
end

#function to compute individual investments
function singlesim(fund, prob, dist, nt, np)
   # matrix to store for each project at any time point the investment made at that time point
   invest_tp = zeros(Float64, np, nt)
   # matrix to store returns
   ret_tp = zeros(Float64, np, nt)
   # vector so store for every project last funding round
   fund_t = zeros(Int, np);

   for t=1:nt
      for p=1:np

         # funding phase
         if fund[p,t] < 0
            d = phase_transition_check( prob[(1+4*(p-1)),t], prob[(2+4*(p-1)),t], prob[(3+4*(p-1)),t], prob[(4+4*(p-1)),t])
            if d==1
               # invest and move to next phase
               invest_tp[p,t] += fund[p,t]
               fund_t[p] = t;
            elseif d==2
               # dont invest, but project continous
            elseif d==0
               # repeat phase
               # first copy last funding value to current position and move the rest forward
               if fund_t[p] > 0
                  fund[p,t:end] = fund[p,fund_t[p]:(end-(t-fund_t[p]))]
                  dist[p,t:end] = dist[p,fund_t[p]:(end-(t-fund_t[p]))]
                  prob[(1+4*(p-1)):(4+4*(p-1)),t:end] = prob[(1+4*(p-1)):(4+4*(p-1)),fund_t[p]:(end-(t-fund_t[p]))]
               end
               # invest and continue
               invest_tp[p,t] += fund[p,t]
               fund_t[p] = t;
            elseif d==3
               # complete write off
               fund[p,t:end] .= 0
               dist[p,t:end] .= 0
               prob[(1+4*(p-1)):(4+4*(p-1)),t:end] .= 0
            end
         end

         # return calculation
         if dist[p,t] > 0
            ret_tp[p,t] += abs(sum(invest_tp[p,:])) * dist[p,t]
         end
      end
   end

   return invest_tp, ret_tp;
end

# function to calculate timespan in fractions of a year (for annualised IRR calculation)
function cf_freq(dates)
    map(d -> DayCounts.yearfrac(dates[1],d,DayCounts.Actual365Fixed()),dates)
end

# function for Net Present Value calculation
function xnpv(xirr,cf,interval)
    sum(cf./(1+xirr).^interval)
end

# function for IRR calculation
function xirr(cf,dates)
    interval = cf_freq(dates)
    f(x) = xnpv(x,cf,interval)
    result = optimize(x -> f(x)^2,0.0,1.0,Brent())
    return result.minimizer
end

# function for Cash Multiple
function mcsim(n_mc, fund, prob, dist)
   nt = size(fund,2)
   np = size(fund,1)

   invest_mc = zeros(Float64, n_mc, nt)
   ret = zeros(Float64, n_mc, nt)
   net_cf = zeros(Float64, n_mc, nt)
   net_cs = zeros(Float64, n_mc, nt)
   irr = zeros(Float64, n_mc)

   dates = Date(2020,1,1):Month(3):Date(2030,12,31)

   for nn=1:n_mc
      a = singlesim( copy(fund), copy(prob), copy(dist), nt, np)
      invest_mc[nn,:] = sum(a[1], dims=1)'
      ret[nn,:] = sum(a[2], dims=1)'
      net_cf[nn,:] = invest_mc[nn,:] + ret[nn,:]
      net_cs[nn,:] = cumsum( sum(a[1], dims=1)' + sum(a[2], dims=1)' , dims=1)
      irr[nn] = xirr( net_cf[nn,:], dates)
   end

   return invest_mc, ret, net_cf, net_cs, irr
end

a = mcsim(100000, fund, prob, dist)

# Generate charts
# Chart: Total invest (Distribution of maximum investment)
histogram(abs.(sum(a[1], dims=2)), fillcolor=cgold, normed=true, title="Distribution of maximum investment", xlab="EUR", ylab="Frequency", lab="", fontfamily="Avantgarde Book", xlims=(2e4,9.5e4), ylims=(0,8e-5))
savefig("Totalinvest.pdf")
# Chart: Total returns (Distribution of Returns)
histogram(sum(a[2], dims=2), fillcolor=cgold, normed=true, title="Distribution of Returns", xlab="EUR", ylab="Frequency", lab="", fontfamily="Avantgarde Book", xlims=(0,4e5), ylims=(0,1.7e-5))
savefig("returns.pdf")
# Chart: Cash Flow (Gross CF at the end)
histogram(a[4][:,end], fillcolor=cgold, normed=true, title="Gross CF at the end", xlab="EUR", ylab="Frequency", lab="", fontfamily="Avantgarde Book", xlims=(0,3.2e5), ylims=(0,2e-5))
savefig("CS.pdf")
# Chart: Multiple (Distribution of Gross Multiple)
histogram((sum(a[2], dims=2)./abs.(sum(a[1], dims=2))), fillcolor=cgold, normed=true, title="Distribution of Gross Multiple", xlab="Multiple", ylab="Frequency", lab="", fontfamily="Avantgarde Book", xlims=(1,6), ylims=(0,1.25))
savefig("multiple.pdf")
#Char Gross IRR
histogram(a[5], fillcolor=cgold, normed=true, title="Gross IRR", xlab="", ylab="Frequency", lab="", fontfamily="Avantgarde Book", xlims=(0.15,0.4), ylims=(0,18))
savefig("IRR.pdf")

# Fund KPI calcutlation
res = zeros(Float64, 5,4)
hh = abs.(sum(a[1], dims=2))
res[1,:] = [mean(hh), median(hh), (mean(hh)+2*std(hh)), (mean(hh)-2*std(hh)) ]
hh = sum(a[2], dims=2)
res[2,:] = [mean(hh), median(hh), (mean(hh)+2*std(hh)), (mean(hh)-2*std(hh)) ]
hh = a[4][:,end]
res[3,:] = [mean(hh), median(hh), (mean(hh)+2*std(hh)), (mean(hh)-2*std(hh)) ]
hh = (sum(a[2], dims=2)./abs.(sum(a[1], dims=2)))
res[4,:] = [mean(hh), median(hh), (mean(hh)+2*std(hh)), (mean(hh)-2*std(hh)) ]
hh = a[5]
res[5,:] = [mean(hh), median(hh), (mean(hh)+2*std(hh)), (mean(hh)-2*std(hh)) ]

res
print(res)

# Write KPIs to CSV
# Matrix(res)
# CSV.write("KPIs.csv", DataFrame(res,auto)
# table = table(res)
# CSV.write(file, table; kwargs...) => file
# table |> CSV.write(file; kwargs...) => file

resss = singlesim( copy(fund), copy(prob), copy(dist), size(fund,2), size(fund,1))

# Chart: Cycle capital calls & distributions
bar(sum(resss[1], dims=1)',lab="Capital Calls", legend=:topleft, xlims = (1,44), fillcolor=crosarot, fontfamily="Avantgarde Book")
xticks!([1:4:44...], string.([2020:2030...]))
bar!(sum(resss[2], dims=1)',lab="Distributions", fillcolor=cgreen)
plot!(cumsum(sum(resss[1], dims=1)'+sum(resss[2], dims=1)', dims=1), lw=4, lab="net CF", linecolor=cgold)
yaxis!("k EUR")
savefig("Cycle.pdf")
