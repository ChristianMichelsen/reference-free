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

# do_GLM = false
# do_lgb_normal = false
# do_evaluate = false
# do_roc = false
# do_shap = false
# do_bases_included_accuracy = false
save_figures = false

#%%

# x = x


# filename_csv = "df__N_reads__100000.csv"
# df = DataFrame(CSV.File(filename_csv, drop = [1]));

filename = "./data/df.data"
N_rows = 1_000_000
# N_rows = 1_000
# N_rows = -1

X, y = get_Xy(filename, N_rows);
train, test = partition(eachindex(y), 0.75; shuffle = true, rng = StableRNG(123));
y_test = y[test];

#%%

x = x

#%%

if false
    println("Signal proportion, base-stratified")
    get_base_stratified_signal_proportion(X[test, :], y_test, mean, Float64)
    get_base_stratified_signal_proportion(X[test, :], y_test, sum, Int64)
end

#%%

if false
    println("Base count fractions")
    mask_signal = (y .== 1)
    base_counts_signal = get_base_counts_pr_position(X[mask_signal, :], normalise = true)
    base_counts_background =
        get_base_counts_pr_position(X[.!mask_signal, :], normalise = true)
    f_base_fraction = make_base_fraction_plot(base_counts_signal, base_counts_background)
    if save_figures
        save("./figures/base_fraction__$(N_rows).pdf", f_base_fraction)
    end
end




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

pipe_logreg = Pipeline(
    OneHotEncoder,
    LogReg(penalty = :none, fit_intercept = false),
    # name = "pipeline_logreg",
    cache = false,
)

mach_logreg = machine(pipe_logreg, X, y)
fit!(mach_logreg, rows = train, verbosity = 0)
yhat_logreg = predict(mach_logreg, rows = test);
acc_logreg = accuracy(predict_mode(mach_logreg, rows = test), y_test)
println("Accuracy, LogReg, ", round(acc_logreg * 100, digits = 2), "%")
confusion_matrix(yhat_logreg, y_test)  # UnivariateFiniteVector

MLJ.save("./data/mach_logreg__$(N_rows).jlso", mach_logreg)

# mach2 = machine("./data/mach_logreg__$(N_rows).jlso")
# predict(mach2, X)
# predict(mach2, rows = test)

df_logreg_long = get_df_logreg_long(mach_logreg);
df_logreg_wide = get_df_logreg_wide(df_logreg_long)
f_LR_fit_coef = plot_LR_fit_coefficients(df_logreg_wide)
if save_figures
    save("./figures/LR_fit_coefficient__$(N_rows).pdf", f_LR_fit_coef)
end

f_density_scores_logreg =
    plot_density_scores(yhat_logreg, y_test, "Density of scores for LR")
if save_figures
    save("./figures/density_scores__LR__$(N_rows).pdf", f_density_scores_logreg)
end

println("AUC, base-stratified, Logistic Regression:")
base_strat_auc_logreg =
    get_base_stratified_measure(X[test, :], yhat_logreg, y_test, area_under_curve)

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



#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                      ██████  ██      ███    ███
#                     ██       ██      ████  ████
#                     ██   ███ ██      ██ ████ ██
#                     ██    ██ ██      ██  ██  ██
#                      ██████  ███████ ██      ██
# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


if do_GLM

    pipe_GLM = Pipeline(
        OneHotEncoder(drop_last = true),
        x -> table(Matrix(x)),
        LinearBinaryClassifier(fit_intercept = true),
        # name = "pipeline_glm",
        cache = false,
    )

    mach_GLM = machine(pipe_GLM, X, y)
    fit!(mach_GLM, rows = train, verbosity = 0)
    fitresult = get_glm_fitresult(mach_GLM)
    yhat_GLM = predict(mach_GLM, rows = test)

    acc_glm = accuracy(predict_mode(mach_GLM, rows = test), y_test)
    println("Accuracy, GLM, ", round(acc_glm, digits = 3))
    confusion_matrix(yhat_GLM, y_test)

    MLJ.save("./data/mach_GLM__$(N_rows).jlso", mach_GLM)

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

    pipe_lgb_normal = Pipeline(
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
        cache = false,
    )

    mach_lgb_normal = machine(pipe_lgb_normal, X, y)
    fit!(mach_lgb_normal, rows = train, verbosity = 0)
    yhat_lgb_normal = predict(mach_lgb_normal, rows = test)
    acc_lgb_normal = accuracy(predict_mode(mach_lgb_normal, rows = test), y_test)
    println("Accuracy, LGB normal, ", round(acc_lgb_normal, digits = 3))
    confusion_matrix(yhat_lgb_normal, y_test)

    MLJ.save("./data/mach_lgb_normal__$(N_rows).jlso", mach_lgb_normal)


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

pipe_lgb_cat = Pipeline(
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
    cache = false,
);
mach_lgb_cat = machine(pipe_lgb_cat, X, y);
fit!(mach_lgb_cat, rows = train, verbosity = 0)
yhat_lgb_cat = predict(mach_lgb_cat, rows = test);

acc_lgb_cat = accuracy(predict_mode(mach_lgb_cat, rows = test), y_test);
println("Accuracy, LGB Cat, ", round(acc_lgb_cat, digits = 3))
confusion_matrix(yhat_lgb_cat, y_test)

MLJ.save("./data/mach_lgb_cat__$(N_rows).jlso", mach_lgb_cat)

f_density_scores_lgb_cat =
    plot_density_scores(yhat_lgb_cat, y_test, "Density of scores for LGB Cat")
if save_figures
    save("./figures/density_scores__LGB_Cat__$(N_rows).pdf", f_density_scores_lgb_cat)
end

println("AUC, base-stratified, LightGBM Categorical:")
base_strat_auc_lgb_cat =
    get_base_stratified_measure(X[test, :], yhat_lgb_cat, y_test, area_under_curve)


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

    # GLMakie.activate!()
    CairoMakie.activate!()
    accuracies = get_accuracies_pr_base(X)
    f_acc = plot_accuracy_function_of_bases(accuracies)
    if save_figures
        save("./figures/accuracies_base_dependent__$(N_rows).pdf", f_acc)
    end

    accuracies_centered = get_accuracies_pr_base_centered(X, add_analytical=true)
    f_acc_centered = plot_accuracy_function_of_bases_centered(accuracies_centered)
    if save_figures
        save("./figures/accuracies_base_dependent_centered__$(N_rows).pdf", f_acc_centered)
    end

    serialize(
        "./data/accuracies__$(N_rows).data",
        (accuracies = accuracies, accuracies_centered = accuracies_centered),
    )

    # accuracies = deserialize("./data/accuracies__$(N_rows).data").accuracies
    # accuracies_centered = deserialize("./data/accuracies__$(N_rows).data").accuracies_centered

    #%%

end

#%%



# half_seq_length = Int((size(X, 2)) / 2)

# middle_idxs = half_seq_length:half_seq_length+1
# middle_X = X[:, middle_idxs]

# seed!(42);
# mach_logreg = machine(pipe_logreg, middle_X, y)
# fit!(mach_logreg, rows = train, verbosity = 0)


# accuracy(predict_mode(mach_logreg, rows = test), y_test)


# XX = DataFrame(x38 = vcat([fill(i, 4) for i = 1:4]...), x39 = repeat(collect(1:4), 4))
# coerce!(XX, :x38 => Multiclass, :x39 => Multiclass)


# predict(mach_logreg, XX)
# predict_mode(mach_logreg, XX)
