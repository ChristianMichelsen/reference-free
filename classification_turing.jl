
using DataFrames
using Serialization
using BioSequences
using MLJ

# permutedims(levels) .== df.x1

#%%

filename_out = "./df.data"
object = deserialize(filename_out)
df = object.df;
df = df[!, 1:1+5];

X_cols = names(df, Not(:y));
df[!, X_cols] = string.(df[!, X_cols])

# schema(df)

#%%


df_c = coerce(df, :y => Binary, [(Symbol(col) => Binary) for col in X_cols]...);
# schema(df_c)

y, X = unpack(df_c, ==(:y), colname -> true; rng = 123);


hot_model = OneHotEncoder()
hot = machine(hot_model, X)
fit!(hot)
Xt = MLJ.transform(hot, X);


# using Plots, LaTeXStrings

# function logistic(x)
#     return 1 / (1 + exp(-x))
# end

# plot(logistic, -10, 10, label = false, xlabel = L"x", ylabel = L"\mathrm{Logistic}(x)")

using Turing
using LazyArrays
using Random: seed!
seed!(123)
# using Memoization
# using ReverseDiff
# using Zygote


# # Set up as Turing model
# Turing.setadbackend(:forwarddiff)
# Turing.setadbackend(:reversediff)
# Turing.setrdcache(true) # fixed size loops and no run-time if statements

@model logreg(X, y; predictors = size(X, 2)) = begin
    #priors
    α ~ Normal(0, 2.5)
    β ~ filldist(TDist(3), predictors)

    #likelihood
    y ~ arraydist(LazyArray(@~ BernoulliLogit.(α .+ X * β)))
end;

# using CSV, HTTP
# url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/wells.csv"
# wells = CSV.read(HTTP.get(url).body, DataFrame)
# describe(wells)

# X = Matrix(select(wells, Not(:switch)))
# y = wells[:, :switch]

XX = Matrix(Xt)
yy = int(y, type = Int) .- 1


using Optim

# Generate a MLE estimate.
model_all = logreg(XX, yy);
mle_estimate = optimize(model_all, MLE())

# Generate a MAP estimate.
map_estimate = optimize(model_all, MAP())

println(map_estimate.values)
# fieldnames(typeof(map_estimate))


# # ADVI
# advi = ADVI(10, 100)
# q = vi(model_all, advi)
# # (mean(rand(q, 1000); dims = 2)..., )



# N = 100
# model = logreg(XX[1:N, :], yy[1:N]);

# using BenchmarkTools
# # using Logging
# # Logging.disable_logging(Logging.Warn)

# chain = sample(model, NUTS(), MCMCThreads(), 2_000, 4)
# # summarystats(chain)
# mean(chain)
# quantile(chain, q = [0.5])


# # plot(chn)


# using Chain

# @chain quantile(chain, q = [0.5]) begin
#     DataFrame
#     select(_, :parameters, names(_, r"%") .=> ByRow(exp), renamecols = false)
# end

# function logodds2prob(logodds::Float64)
#     return exp(logodds) / (1 + exp(logodds))
# end

# @chain quantile(chain, q = [0.5]) begin
#     DataFrame
#     select(_, :parameters, names(_, r"%") .=> ByRow(logodds2prob), renamecols = false)
# end

