
using DataFrames
using Serialization
using BioSequences
using MLJ
using MLJGLMInterface

#%%

filename_out = "./df.data"
object = deserialize(filename_out);
df = object.df;
df.y = Int64.(df.y)
df = df[!, 1:1+15];

X_cols = names(df, Not(:y));
df[!, X_cols] = string.(df[!, X_cols]);

encoding = autotype(df, :string_to_multiclass)
df_c = coerce(df, :y => Binary, encoding...)


X = select(df_c, Not(:y))
y = df_c.y

#%%

function get_matching_models(df)
    hot_model = OneHotEncoder()
    hot = machine(hot_model, X)
    fit!(hot)
    Xt = MLJ.transform(hot, X)
    # models(matching(X, y))
    # models(matching(Xt, y))
    models() do model
        matching(model, Xt, y) && model.prediction_type == :probabilistic #&&
        # model.is_pure_julia
    end
end
get_matching_models(df_c)

#%%


LogReg = @load LogisticClassifier pkg = MLJLinearModels

pipe_logreg = @pipeline(
    OneHotEncoder,
    LogReg(penalty = :none, fit_intercept = true),
    name = "pipeline_logreg",
)

mach_logreg = machine(pipe_logreg, X, y)
fit!(mach_logreg)
yhat_logreg = predict(mach_logreg, X)
acc_logreg = accuracy(predict_mode(mach_logreg, X), y)
println("Accuracy, LogReg, ", round(acc_logreg, digits = 3))


evaluate(
    pipe_logreg,
    X,
    y,
    resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
    measures = [LogLoss(), Accuracy()],
    verbosity = 0,
)

function transpose(df::DataFrame)
    df_T = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
end

function get_logreg_dataframe(mach::Machine)
    params_raw = fitted_params(mach_logreg)
    params = hcat(
        DataFrame(intercept = params_raw.logistic_classifier.intercept),
        DataFrame(params_raw.logistic_classifier.coefs),
    )

    params_T = rename(transpose(params), :2 => :values)
    return params_T
end
get_logreg_dataframe(mach_logreg)

#%%


# evaluate(
#     pipe_logreg,
#     X,
#     y,
#     resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
#     measures = [LogLoss(), Accuracy()],
#     verbosity = 0,
# )


#%%

# GLM


pipe_GLM = @pipeline(
    OneHotEncoder(drop_last = true),
    x -> table(Matrix(x)),
    LinearBinaryClassifier(),
    name = "pipeline_glm",
)

mach_logregGLM = machine(pipe_GLM, X, y);
fit!(mach_logregGLM);

function get_glm_fitresult(mach::Machine)
    fitresult, decode = mach.fitresult.predict.machine.fitresult
    return fitresult
end
fitresult = get_glm_fitresult(mach_logregGLM)
yhat_GLM = predict(mach_logregGLM, X);

acc_glm = accuracy(predict_mode(mach_logregGLM, X), y)
println("Accuracy, GLM, ", round(acc_glm, digits = 3))


# confusion_matrix(categorical([1, 1, 0]; ordered=true), categorical([0, 1, 0]; ordered=true))




fitted_params(mach_logregGLM).linear_binary_classifier.coef
r = report(mach_logregGLM).linear_binary_classifier;
r.stderror
# r.deviance
# r.dof_residual
# r.vcov


# pdf.(yhat_logreg, 1)


#%%

# XGB

XGB = @load XGBoostClassifier
pipe_xgb = @pipeline(OneHotEncoder, XGB(), name = "pipeline_xgb",);
machine_xgb = machine(pipe_xgb, X, y);
fit!(machine_xgb)
yhat_xgb = predict(machine_xgb, X);
# predict_mean(machine_xgb, X)
acc_xgb = accuracy(predict_mode(machine_xgb, X), y)
println("Accuracy, XGB, ", round(acc_xgb, digits = 3))



# LGB

LGB = @load LGBMClassifier
pipe_lgb = @pipeline(
    OneHotEncoder,
    LGB(
        objective = "binary",
        num_iterations = 100,
        learning_rate = 0.1,
        # early_stopping_round = 5,
        feature_fraction = 0.8,
        bagging_fraction = 0.9,
        bagging_freq = 1,
        num_leaves = 1000,
        # num_class = 1,
        metric = ["auc", "binary_logloss"],
    ),
    # name = "pipeline_lgb",
);
machine_lgb = machine(pipe_lgb, X, y);
fit!(machine_lgb)
yhat_lgb = predict(machine_lgb, X);
acc_lgb = accuracy(predict_mode(machine_lgb, X), y)
println("Accuracy, LGB, ", round(acc_lgb, digits = 3))


evaluate(
    pipe_lgb,
    X,
    y,
    resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
    measures = [LogLoss(), Accuracy()],
    verbosity = 0,
)