# using DataFrames
# using Serialization
# using BioSequences
# using MLJ
using MLJGLMInterface
using StableRNGs
# using CSV

#%%

include("mlj_functions.jl")

#%%

do_GLM = true
do_evaluate = true
do_shap = true
do_bases_included_accuracy = true
make_figures = true
save_figures = true

do_GLM = false
do_evaluate = false
do_shap = false
# do_bases_included_accuracy = false
make_figures = false
save_figures = false

#%%

# x = x

# filename_csv = "df__N_reads__100000.csv"
# df = DataFrame(CSV.File(filename_csv, drop = [1]));

filename = "./data/df.data"
N_rows = 1_000_000
# N_rows = 10_000

X, y = get_Xy(filename, N_rows);
train, test = partition(eachindex(y), 0.75; shuffle = true, rng = StableRNG(123));

X_train = selectrows(X, train);
y_train = selectrows(y, train);
X_test = selectrows(X, test);
y_test = selectrows(y, test);

#%%

# x = x

#%%

if make_figures
    println("Signal proportion, base-stratified")
    get_base_stratified_signal_proportion(X[test, :], y_test, mean, Float64)
    get_base_stratified_signal_proportion(X[test, :], y_test, sum, Int64)
end

#%%

if make_figures
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


LR = @load LogisticClassifier pkg = MLJLinearModels

pipe_LR = Pipeline(OneHotEncoder, LR(penalty = :none, fit_intercept = false), cache = false);

filename_LR = "./data/mach_LR__$(N_rows).jlso"

if isfile(filename_LR)
    println("Loading LR: $filename_LR")
    mach_LR = machine(filename_LR)
else
    println("Fitting LR: $filename_LR")
    mach_LR = machine(pipe_LR, X_train, y_train)
    fit!(mach_LR, verbosity = 0)
    MLJ.save(filename_LR, mach_LR)

    df_LR_long = get_df_LR_long(mach_LR)
    df_LR_wide = get_df_LR_wide(df_LR_long)
    if make_figures
        f_LR_fit_coef = plot_LR_fit_coefficients(df_LR_wide)
        if save_figures
            save("./figures/LR_fit_coefficient__$(N_rows).pdf", f_LR_fit_coef)
        end
    end

end


yhat_LR = predict(mach_LR, X_test);
acc_LR = accuracy(yhat_LR, y_test)
println("Accuracy, LR, ", round(acc_LR * 100, digits = 2), "%")
confusion_matrix(yhat_LR, y_test)

if make_figures
    f_score_density_LR = plot_score_density(yhat_LR, y_test, "Density of scores for LR")
    if save_figures
        save("./figures/score_density__LR__$(N_rows).pdf", f_score_density_LR)
    end
end

println("AUC, base-stratified, Logistic Regression:")
base_strat_auc_LR = get_base_stratified_measure(X_test, yhat_LR, y_test, area_under_curve)

if do_evaluate
    eval_LR = evaluate(
        pipe_LR,
        X,
        y,
        resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
        measures = [LogLoss(), Accuracy()],
        verbosity = 0,
    )

    print_performance(eval_LR, "Logistic Regression")
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
        cache = false,
    )

    filename_GLM = "./data/mach_GLM__$(N_rows).jlso"

    if isfile(filename_GLM)
        println("Loading GLM: $filename_GLM")
        mach_GLM = machine(filename_GLM)
    else
        println("Fitting GLM: $filename_GLM")
        mach_GLM = machine(pipe_GLM, X_train, y_train)
        fit!(mach_GLM, verbosity = 0)
        MLJ.save(filename_GLM, mach_GLM)

        fitresult = get_glm_fitresult(mach_GLM)

        fitted_params(mach_GLM).linear_binary_classifier.coef
        r = report(mach_GLM).linear_binary_classifier
        r.stderror
        # r.deviance
        # r.dof_residual
        # r.vcov
    end

    yhat_GLM = predict(mach_GLM, X_test)

    acc_glm = accuracy(yhat_GLM, y_test)
    println("Accuracy, GLM, ", round(100 * acc_glm, digits = 2), "%")
    confusion_matrix(yhat_GLM, y_test)

end

#%% LGB categorical
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██       ██████  ██████   ██████
#                     ██      ██       ██   ██ ██
#                     ██      ██   ███ ██████  ██
#                     ██      ██    ██ ██   ██ ██
#                     ███████  ██████  ██████   ██████
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

LGB = @load LGBMClassifier verbosity = 0
categorical_columns = collect(1:size(X)[2]);

pipe_LGB = Pipeline(
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
    cache = false,
);


filename_LGB = "./data/mach_LGB__$(N_rows).jlso"

if isfile(filename_LGB)
    println("Loading LGB: $filename_LGB")
    mach_LGB = machine(filename_LGB)
else
    println("Fitting LGB: $filename_LGB")
    mach_LGB = machine(pipe_LGB, X_train, y_train)
    fit!(mach_LGB, verbosity = 0)
    MLJ.save(filename_LGB, mach_LGB)
end


yhat_LGB = predict(mach_LGB, X_test);
acc_LGB = accuracy(yhat_LGB, y_test);
println("Accuracy, LGB, ", round(100 * acc_LGB, digits = 2), "%")
confusion_matrix(yhat_LGB, y_test)

if make_figures
    f_score_density_LGB = plot_score_density(yhat_LGB, y_test, "Density of scores for LGB")
    if save_figures
        save("./figures/score_density__LGB__$(N_rows).pdf", f_score_density_LGB)
    end
end

println("AUC, base-stratified, LightGBM:")
base_strat_auc_LGB = get_base_stratified_measure(X_test, yhat_LGB, y_test, area_under_curve)


if do_evaluate
    eval_LGB = evaluate(
        pipe_LGB,
        X,
        y,
        resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
        measures = [LogLoss(), Accuracy()],
        verbosity = 0,
    )
    print_performance(eval_LGB, "LightGBM")
end


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
# mach = machine(pipe_LGB2, X, y)
# boostrange = range(ensemble, :(LGBm_classifier.num_iterations), lower=2, upper=500)
# curve = learning_curve!(mach, resampling=CV(nfolds=5),
#                         range=boostrange,
#                         resolution=100,
#                         measure=LogLoss())

# mach = fit!(machine(OneHotEncoder(), X))
# Xt = MLJ.transform(mach, X)


# LGB = LGB(); #initialised a model with default params
# LGBm = machine(LGB, Xt, y)
# boostrange = range(LGB, :num_iterations, lower=2, upper=500)
# curve = learning_curve!(LGBm, resampling=CV(nfolds=5),
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


if make_figures

    y_hats = (yhat_LR, yhat_LGB)
    names_roc = ("Logistic Regression", "LightGBM")
    f_roc = plot_roc(y_hats, y_test, names_roc)
    if save_figures
        save("./figures/ROC__$(N_rows).pdf", f_roc)
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

    data_shap_LR = shap_compute(mach = mach_LR, X = X_test, N = 100)

    if make_figures
        plot_shap_global(data_shap_LR, savefig = true, filename = "shap_global_LR")
        plot_shap_variables(
            data_shap_LR,
            savefig = save_figures,
            filename = "shap_feature_LR",
        )
    end

    data_shap_LGB = shap_compute(mach = mach_LGB, X = X_test, N = 100)
    if make_figures
        plot_shap_global(
            data_shap_LGB,
            savefig = save_figures,
            filename = "shap_global_LGB",
        )
        plot_shap_variables(
            data_shap_LGB,
            savefig = save_figures,
            filename = "shap_feature_LGB",
        )
    end

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

    filename_accuracies = "./data/accuracies__$(N_rows).data"

    if isfile(filename_accuracies)
        accuracies = deserialize(filename_accuracies).accuracies
        accuracies_centered = deserialize(filename_accuracies).accuracies_centered
    else
        accuracies = get_accuracies_pr_base(X, y, train, test)
        accuracies_centered = get_accuracies_pr_base_centered(
            X,
            y,
            train,
            test;
            add_analytical = true,
            add_custom = true,
        )

        serialize(
            filename_accuracies,
            (accuracies = accuracies, accuracies_centered = accuracies_centered),
        )

    end

    if make_figures
        # GLMakie.activate!()
        CairoMakie.activate!()

        f_acc = plot_accuracy_function_of_bases(accuracies)
        if save_figures
            save("./figures/accuracies_base_dependent__$(N_rows).pdf", f_acc)
        end

        f_acc_centered = plot_accuracy_function_of_bases_centered(accuracies_centered)
        if save_figures
            save(
                "./figures/accuracies_base_dependent_centered__$(N_rows).pdf",
                f_acc_centered,
            )
        end
    end


    #%%

end

#%%

