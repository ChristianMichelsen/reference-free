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

N_rows = 10000
X, y = get_data(N_rows);

LogReg = @load LogisticClassifier pkg = MLJLinearModels
pipe_logreg = @pipeline(OneHotEncoder, LogReg(penalty = :none, fit_intercept = false),)

mach = machine(pipe_logreg, X, y)
fit!(mach)
MLJ.save("pipeline_test_$N_rows.jlso", mach)

fitted_params(mach).logistic_classifier