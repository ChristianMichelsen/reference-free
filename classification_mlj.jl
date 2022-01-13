using DataFrames
using Serialization
using BioSequences
using MLJ
using MLJGLMInterface
using StableRNGs
using StatsBase: sample
# using CSV

#%%

include("mlj_functions.jl")
# include("mlj_types_test.jl")

#%%

do_GLM = true
do_lgb_normal = true
do_evaluate = true
do_roc = true
do_shap = false
do_bases_included_accuracy = true
save_figures = true

#%%

# x = x


# filename_csv = "df__N_reads__100000.csv"
# df = DataFrame(CSV.File(filename_csv, drop = [1]));

filename = "./df.data"
N_rows = 1_000_000
# N_rows = -1

X, y = get_Xy(filename, N_rows);
train, test = partition(eachindex(y), 0.75; shuffle = true, rng = StableRNG(123));
y_test = y[test];


# x = x


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
println("Accuracy, LogReg, ", round(acc_logreg * 100, digits = 2), "%")
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
if save_figures
    save("./figures/LR_fit_coefficient__$(N_rows).pdf", f_LR_fit_coef)
end


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

if do_lgb_normal

    pipe_lgb_normal = @pipeline(
        OneHotEncoder,
        LGB(
            objective = "binary",
            num_iterations = 100,
            learning_rate = 0.1,
            feature_fraction = 0.8,
            bagging_fraction = 0.9,
            bagging_freq = 1,
            num_leaves = 1000,
            metric = ["auc", "binary_logloss"],
        ),
        # name = "pipeline_lgb_normal",
    )

    mach_lgb_normal = machine(pipe_lgb_normal, X, y)
    fit!(mach_lgb_normal, rows = train, verbosity = 0)
    yhat_lgb_normal = predict(mach_lgb_normal, rows = test)
    acc_lgb_normal = accuracy(predict_mode(mach_lgb_normal, rows = test), y_test)
    println("Accuracy, LGB normal, ", round(acc_lgb_normal, digits = 3))
    confusion_matrix(yhat_lgb_normal, y_test)

    if do_evaluate
        eval_lgb_normal = evaluate(
            pipe_lgb_normal,
            X,
            y,
            resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
            measures = [LogLoss(), Accuracy()],
            verbosity = 0,
        )

        print_performance(eval_lgb_normal, "LightGBM")
    end

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
    x -> convert_type(x, Float64),
    LGB(
        objective = "binary",
        num_iterations = 100,
        learning_rate = 0.1,
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


#%%

# x = x

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


if do_roc

    y_hats = (yhat_logreg, yhat_lgb_cat)
    names_roc = ("Logistic Regression", "LightGBM Categorical")
    f_roc = plot_roc(y_hats, y_test, names_roc)
    if save_figures
        save("./figures/fig_ROC__$(N_rows).pdf", f_roc)
    end

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
        savefig = save_figures,
        filename = "fig_shap_feature_logreg",
    )


    data_shap_lgb = shap_compute(mach = mach_lgb, X = X[test, :], N = 100)
    plot_shap_global(data_shap_lgb, savefig = true, filename = "fig_shap_global_lgb")
    plot_shap_variables(data_shap_lgb, savefig = true, filename = "fig_shap_feature_lgb")


    data_shap_lgb_cat = shap_compute(mach = mach_lgb_cat, X = X[test, :], N = 100)
    plot_shap_global(
        data_shap_lgb_cat,
        savefig = save_figures,
        filename = "fig_shap_global_lgb_cat",
    )
    plot_shap_variables(
        data_shap_lgb_cat,
        savefig = save_figures,
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

    accuracies = get_accuracies_pr_base(X)

    # GLMakie.activate!()
    CairoMakie.activate!()

    # f_acc = plot_accuracy_function_of_bases(accuracies, ylimits = (0.634, 0.701))
    f_acc = plot_accuracy_function_of_bases(accuracies)
    if save_figures
        save("./figures/accuracies_base_dependent__$(N_rows).pdf", f_acc)
    end

end

#%%


