I am running a MLJ pipeline on a quite large dataset and I want to be able to save the model once, and then use it in later sessions without having to retrain it. I have tried following the [documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/machines/#Saving-machines-1) regarding saving models and that works fine. However, I am quite puzzled at the fact that the size of the saved model changes depending on the size of the dataset.

A MWE below:

```julia
using DataFrames
using MLJ

function get_data(N_rows)
    df = DataFrame(rand(N_rows, 3), :auto)
    df.x4 = rand(["A", "B", "C"], N_rows)
    df.y = rand([0, 1], N_rows)
    df = coerce(df, :y => Binary, :x4 => Multiclass)

    X = select(df, Not(:y))
    y = df.y
    return X, y
end

N_rows = 1000
X, y = get_data(N_rows);

LogReg = @load LogisticClassifier pkg = MLJLinearModels
pipe_logreg = @pipeline(OneHotEncoder, LogReg(penalty = :none, fit_intercept = false),)

mach = machine(pipe_logreg, X, y)
fit!(mach)
MLJ.save("pipeline_test_$N_rows.jlso", mach)
```
The size of the saved model is 227 KB when `N_rows = 1_000` and 1.7 MB when `N_rows = 10_000`, yet they both save the same number of parameters (3 numerical coefficients and 3 coefficients for the categories). Or, that's at least what I expect, but why does the size really change so drastically with more data? I do not expect MLJ to save the input data as well as that would be quite unintuitive.

This is run with Julia v1.7.0, MLJ v0.16.11, MLJBase v0.18.26,  and MLJLinearModels v0.5.7.