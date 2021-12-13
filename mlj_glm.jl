using DataFrames

import Distributions
import StableRNGs
using MLJGLMInterface
using MLJBase
import GLM
using MLJ

rng = StableRNGs.StableRNG(0)
N = 1000
X = DataFrame(x1 = categorical(rand([1, 2, 2, 3], N)));
ycont = 0.7 * Int.(X.x1 .== 2) + rand(rng, N);
y = categorical(ycont .> mean(ycont));

formula = GLM.@formula(y ~ 1 + x1)
df = hcat(X, DataFrame(y = int(y, type = Int) .- 1));
logistic = GLM.glm(formula, df, GLM.Bernoulli(), GLM.LogitLink())


pipe = @pipeline(
    OneHotEncoder(drop_last = true),
    x -> MLJBase.table(Matrix(x)),
    LinearBinaryClassifier(),
)

mach = machine(pipe, X, y);
fit!(mach);

function get_glm_fitresult(mach::Machine)
    fitresult, decode = mach.fitresult.predict.machine.fitresult
    return fitresult
end
fitresult = get_glm_fitresult(mach)
yhat = predict(mach, X);


fitted_params(mach).linear_binary_classifier.coef
r = report(mach).linear_binary_classifier;
r.stderror
# r.deviance
# r.dof_residual
# r.vcov


#%%
# coerce(df_X, autotype(X, :string_to_multiclass))


# hot_model = OneHotEncoder(drop_last = true);
# hot = machine(hot_model, X);
# fit!(hot);
# Xt = MLJ.transform(hot, X);

# lr = LinearBinaryClassifier()
# fitresult, _, reportt = fit(lr, 1, MLJBase.table(Matrix(X)), y)
# # yhat = predict(lr, fitresult, X)

# lr = LinearBinaryClassifier()
# fitresult, _, reportt = fit(lr, 1, MLJBase.table(Matrix(Xt)), y)
