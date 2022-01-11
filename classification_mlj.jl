using DataFrames
using Serialization
using BioSequences
using MLJ
using MLJGLMInterface
using StableRNGs
using CSV

#%%

include("mlj_functions.jl")

#%%

do_GLM = false
do_evaluate = true
do_roc = false
do_shap = false
do_bases_included_accuracy = false

#%%


filename_csv = "df__N_reads__100000.csv"

df = DataFrame(CSV.File(filename_csv, drop = [1]));
# filename_out = "./df.data"
# df = get_data(filename_out; N_positions = N_positions, k = k)

X, y = get_X_y(df);
train, test = partition(eachindex(y), 0.75; shuffle = true, rng = StableRNG(123))
y_test = y[test]

x = x


#%%

# get_matching_models(X, y)

# #%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██      ██████
#                     ██      ██   ██
#                     ██      ██████
#                     ██      ██   ██
#                     ███████ ██   ██
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


LogReg = @load LogisticClassifier pkg = MLJLinearModels

pipe_logreg = @pipeline(
    OneHotEncoder,
    LogReg(penalty = :none, fit_intercept = false),
    # name = "pipeline_logreg",
)

mach_logreg = machine(pipe_logreg, X, y)
fit!(mach_logreg, rows = train, verbosity = 0)
yhat_logreg = predict(mach_logreg, rows = test);
acc_logreg = accuracy(predict_mode(mach_logreg, rows = test), y_test)
println("Accuracy, LogReg, ", round(acc_logreg, digits = 3))
confusion_matrix(yhat_logreg, y_test)

if do_evaluate
    eval_logreg = evaluate(
        pipe_logreg,
        X,
        y,
        resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
        measures = [LogLoss(), Accuracy()],
        verbosity = 0,
    )

    print_performance(eval_logreg, "Logistic Regression")
end


df_logreg_long = get_df_logreg_long(mach_logreg);
df_logreg_wide = get_df_logreg_wide(df_logreg_long)
f_LR_fit_coef = plot_LR_fit_coefficients(df_logreg_wide)
save("./figures/LR_fit_coefficient.pdf", f_LR_fit_coef)


#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                      ██████  ██      ███    ███
#                     ██       ██      ████  ████
#                     ██   ███ ██      ██ ████ ██
#                     ██    ██ ██      ██  ██  ██
#                      ██████  ███████ ██      ██
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if do_GLM

    pipe_GLM = @pipeline(
        OneHotEncoder(drop_last = true),
        x -> table(Matrix(x)),
        LinearBinaryClassifier(fit_intercept = true),
        # name = "pipeline_glm",
    )

    mach_GLM = machine(pipe_GLM, X, y)
    fit!(mach_GLM, rows = train, verbosity = 0)
    fitresult = get_glm_fitresult(mach_GLM)
    yhat_GLM = predict(mach_GLM, rows = test)

    acc_glm = accuracy(predict_mode(mach_GLM, rows = test), y_test)
    println("Accuracy, GLM, ", round(acc_glm, digits = 3))
    confusion_matrix(yhat_GLM, y_test)


    fitted_params(mach_GLM).linear_binary_classifier.coef
    r = report(mach_GLM).linear_binary_classifier
    r.stderror
    # r.deviance
    # r.dof_residual
    # r.vcov

end

# LGB
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██       ██████  ██████
#                     ██      ██       ██   ██
#                     ██      ██   ███ ██████
#                     ██      ██    ██ ██   ██
#                     ███████  ██████  ██████
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


LGB = @load LGBMClassifier verbosity = 0

lgb = LGB(
    objective = "binary",
    num_iterations = 100,
    learning_rate = 0.1,
    # early_stopping_round = 5,
    feature_fraction = 0.8,
    bagging_fraction = 0.9,
    bagging_freq = 1,
    num_leaves = 1000,
    metric = ["auc", "binary_logloss"],
);

pipe_lgb = @pipeline(
    OneHotEncoder,
    lgb,
    # name = "pipeline_lgb",
);
mach_lgb = machine(pipe_lgb, X, y);
fit!(mach_lgb, rows = train, verbosity = 0)
yhat_lgb = predict(mach_lgb, rows = test);
acc_lgb = accuracy(predict_mode(mach_lgb, rows = test), y_test)
println("Accuracy, LGB, ", round(acc_lgb, digits = 3))
confusion_matrix(yhat_lgb, y_test)

if do_evaluate
    eval_lgb = evaluate(
        pipe_lgb,
        X,
        y,
        resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
        measures = [LogLoss(), Accuracy()],
        verbosity = 0,
    )

    print_performance(eval_lgb, "LightGBM")
end


#%% LGB categorical
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██       ██████  ██████   ██████
#                     ██      ██       ██   ██ ██
#                     ██      ██   ███ ██████  ██
#                     ██      ██    ██ ██   ██ ██
#                     ███████  ██████  ██████   ██████
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



categorical_columns = collect(1:size(X)[2]);

pipe_lgb_cat = @pipeline(
    X -> convert_type(X, Float64),
    LGB(
        objective = "binary",
        num_iterations = 100,
        learning_rate = 0.1,
        # early_stopping_round = 5,
        feature_fraction = 0.8,
        bagging_fraction = 0.9,
        bagging_freq = 1,
        num_leaves = 1000,
        metric = ["auc", "binary_logloss"],
        categorical_feature = copy(categorical_columns),
    ),
    # yhat -> mode.(yhat)
    # name = "pipeline_lgb",
);
mach_lgb_cat = machine(pipe_lgb_cat, X, y);
fit!(mach_lgb_cat, rows = train, verbosity = 0)
yhat_lgb_cat = predict(mach_lgb_cat, rows = test);

acc_lgb_cat = accuracy(predict_mode(mach_lgb_cat, rows = test), y_test);
println("Accuracy, LGB Cat, ", round(acc_lgb_cat, digits = 3))
confusion_matrix(yhat_lgb_cat, y_test)

if do_evaluate

    eval_lgb_cat = evaluate(
        pipe_lgb_cat,
        X,
        y,
        resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
        measures = [LogLoss(), Accuracy()],
        verbosity = 0,
    )

    print_performance(eval_lgb_cat, "LightGBM Categorical")

end

# #%%

# import LightGBM

# indices = [1, 3, 5, 7, 9]
# classifier = LightGBM.LGBMClassification(categorical_feature = indices)
# ds_parameters = LightGBM.stringifyparams(classifier; verbosity = -1)

# expected = "categorical_feature=0,2,4,6,8"
# occursin(expected, ds_parameters)

#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██   ██ ██████   ██████
#                     ██   ██ ██   ██ ██    ██
#                     ███████ ██████  ██    ██
#                     ██   ██ ██      ██    ██
#                     ██   ██ ██       ██████
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# atom = LGB(); #initialised a model with default params
# ensemble = @pipeline(OneHotEncoder, atom);
# mach = machine(pipe_lgb2, X, y)
# boostrange = range(ensemble, :(lgbm_classifier.num_iterations), lower=2, upper=500)
# curve = learning_curve!(mach, resampling=CV(nfolds=5),
#                         range=boostrange,
#                         resolution=100,
#                         measure=LogLoss())

# mach = fit!(machine(OneHotEncoder(), X))
# Xt = MLJ.transform(mach, X)


# lgb = LGB(); #initialised a model with default params
# lgbm = machine(lgb, Xt, y)
# boostrange = range(lgb, :num_iterations, lower=2, upper=500)
# curve = learning_curve!(lgbm, resampling=CV(nfolds=5),
#                         range=boostrange, resolution=100,
#                         measure=LogLoss())


# x = x

#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██████   ██████   ██████
#                     ██   ██ ██    ██ ██
#                     ██████  ██    ██ ██
#                     ██   ██ ██    ██ ██
#                     ██   ██  ██████   ██████
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


using GLMakie
using CairoMakie

if do_roc

    y_hats = (yhat_logreg, yhat_lgb, yhat_lgb_cat)
    names_roc = ("Logistic Regression", "LightGBM OneHotEncoding", "LightGBM Categorical")
    f_roc = plot_roc(y_hats, y_test, names_roc)
    save("./figures/fig_ROC.pdf", f_roc)

end

# x=x

#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ███████ ██   ██  █████  ██████
#                     ██      ██   ██ ██   ██ ██   ██
#                     ███████ ███████ ███████ ██████
#                          ██ ██   ██ ██   ██ ██
#                     ███████ ██   ██ ██   ██ ██
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if do_shap

    CairoMakie.activate!()

    data_shap_logreg = shap_compute(mach = mach_logreg, X = X[test, :], N = 100)
    plot_shap_global(data_shap_logreg, savefig = true, filename = "fig_shap_global_logreg")
    plot_shap_variables(
        data_shap_logreg,
        savefig = true,
        filename = "fig_shap_feature_logreg",
    )


    data_shap_lgb = shap_compute(mach = mach_lgb, X = X[test, :], N = 100)
    plot_shap_global(data_shap_lgb, savefig = true, filename = "fig_shap_global_lgb")
    plot_shap_variables(data_shap_lgb, savefig = true, filename = "fig_shap_feature_lgb")


    data_shap_lgb_cat = shap_compute(mach = mach_lgb_cat, X = X[test, :], N = 100)
    plot_shap_global(
        data_shap_lgb_cat,
        savefig = true,
        filename = "fig_shap_global_lgb_cat",
    )
    plot_shap_variables(
        data_shap_lgb_cat,
        savefig = true,
        filename = "fig_shap_feature_lgb_cat",
    )

end


#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██████   █████  ███████ ███████ ███████
#                     ██   ██ ██   ██ ██      ██      ██
#                     ██████  ███████ ███████ █████   ███████
#                     ██   ██ ██   ██      ██ ██           ██
#                     ██████  ██   ██ ███████ ███████ ███████
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if do_bases_included_accuracy

    accuracies = get_accuracies_pr_base(df)

    # GLMakie.activate!()
    CairoMakie.activate!()

    f_acc = plot_accuracy_function_of_bases(accuracies, ylimits = (0.634, 0.701))
    save("./figures/accuracies_base_dependent.pdf", f_acc)

end

#%%

