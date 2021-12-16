import MLJBase
using DataFramesMeta
using ShapML
using OrderedCollections



function get_data(filename; N_positions = 15)
    object = deserialize(filename_out)
    df = object.df
    df.y = Int64.(df.y)
    df = df[!, 1:1+N_positions]
    X_cols = names(df, Not(:y))
    df[!, X_cols] = string.(df[!, X_cols])
    return df
end


function get_X_y(df)
    encoding = autotype(df, :string_to_multiclass)
    df_c = coerce(df, :y => Binary, encoding...)
    X = select(df_c, Not(:y))
    y = df_c.y
    return X, y
end


function get_matching_models(X, y)
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

#%%

function transpose(df::DataFrame)
    df_T = DataFrame([[names(df)]; collect.(eachrow(df))], [:column; Symbol.(axes(df, 1))])
    return df_T
end

function get_df_logreg_long(mach::Machine)
    params_raw = fitted_params(mach)
    params = hcat(
        # DataFrame(intercept = params_raw.logistic_classifier.intercept),
        DataFrame(params_raw.logistic_classifier.coefs),
    )
    params_T = rename(transpose(params), :2 => :values)
    return params_T
end


function get_df_logreg_wide(df::DataFrame)
    m = permutedims(reshape(df.values, (4, N_positions)))
    return DataFrame(m, [:A, :C, :G, :T])
end

#%%

function logodds2prob(logodds::Float64)
    return exp(logodds) / (1 + exp(logodds))
end


#%%


function get_glm_fitresult(mach::Machine)
    fitresult, decode = mach.fitresult.predict.machine.fitresult
    return fitresult
end

#%%


#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ██   ██  ██████  ██████
#                      ██ ██  ██       ██   ██
#                       ███   ██   ███ ██████
#                      ██ ██  ██    ██ ██   ██
#                     ██   ██  ██████  ██████
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# XGB = @load XGBoostClassifier
# pipe_xgb = @pipeline(OneHotEncoder, XGB(), name = "pipeline_xgb",);
# machine_xgb = machine(pipe_xgb, X, y);
# fit!(machine_xgb)
# yhat_xgb = predict(machine_xgb, X);
# # predict_mean(machine_xgb, X)
# acc_xgb = accuracy(predict_mode(machine_xgb, X), y)
# println("Accuracy, XGB, ", round(acc_xgb, digits = 3))

# evaluate(
#     pipe_xgb,
#     X,
#     y,
#     resampling = CV(nfolds = 5, shuffle = true, rng = 1234),
#     measures = [LogLoss(), Accuracy()],
#     verbosity = 0,
# )

#%%

function convert_type(df::DataFrame, type::DataType)
    mapcols(col -> int(col; type = type) .- 1, df)
end

#%%

function MLJBase.confusion_matrix(
    yhat::MLJBase.UnivariateFiniteVector,
    y::MLJ.CategoricalVector,
)
    confusion_matrix(coerce(mode.(yhat), OrderedFactor), coerce(y, OrderedFactor))
end


#%%


function is_nested(x)
    if eltype(x) <: AbstractArray
        return true
    else
        return false
    end
end

#%%

function _plot_roc_check(y_hats, names_roc)
    if length(y_hats) != length(names_roc)
        throw(ArgumentError("'y_hats' and 'names_roc' is not the same length."))
    end
    if !is_nested(y_hats)
        y_hats = (y_hats,)
    end
    if isa(names_roc, String)
        names_roc = (names_roc,)
    end
    return y_hats, names_roc
end

function plot_roc(y_hats, y, names_roc)
    y_hats, names_roc = _plot_roc_check(y_hats, names_roc)

    f = Figure()
    ax = Axis(f[1, 1], xlabel = "FPR", ylabel = "TPR", title = "ROC Curve")
    for (y_hat, name) in zip(y_hats, names_roc)
        fprs, tprs, ts = roc_curve(y_hat, y)
        lines!(ax, fprs, tprs, label = name)
    end
    axislegend(ax, position = :rb)
    f
end


#%%

# A wrapper function for ShapML.
function shap_predict_function(model, data)
    y_pred = int(predict_mode(model, data), type = Int64) .- 1
    data_pred = DataFrame(y_pred = y_pred)
    return data_pred
end

function shap_compute(; mach, X, N = 100, sample_size = 100, use_reference = false)

    # Compute Shapley feature-level predictions for N instances.
    explain = copy(X[1:N, :])

    # An optional reference population to compute the baseline prediction.
    reference = use_reference ? copy(X) : nothing

    # sample_size  # Number of Monte Carlo samples.

    # Compute stochastic Shapley values.
    data_shap = ShapML.shap(
        explain = explain,
        reference = reference,
        model = mach,
        predict_function = shap_predict_function,
        sample_size = sample_size,
        seed = 1,
    )

    return data_shap

end

#%%


function get_shap_global_plot_data(data_shap; sort = false)

    data_plot = combine(
        groupby(data_shap, :feature_name),
        [:shap_effect] => (x -> mean(abs.(x))) => :mean_effect,
        [:shap_effect] => (x -> std(abs.(x))) => :std_effect,
        [:shap_effect] => (x -> std(abs.(x)) / sqrt(length(x))) => :sdom_effect,
    )

    if sort
        data_plot = sort(data_plot, order(:mean_effect, rev = true))
    end

    baseline = round(data_shap.intercept[1], digits = 1)

    tbl = (
        x = 1:size(data_plot)[1],
        y = data_plot.mean_effect,
        y_std = data_plot.std_effect,
        y_sdom = data_plot.sdom_effect,
        name = data_plot.feature_name,
        baseline = baseline,
        df = data_plot,
    )

    return tbl
end


function plot_shap_global(
    data_shap;
    sort = false,
    add_errorbars = false,
    savefig = true,
    filename = "fig_shap_global",
)

    tbl = get_shap_global_plot_data(data_shap; sort = sort)

    f = Figure()
    ax = Axis(
        f[1, 1],
        yticks = (tbl.x, tbl.name),
        title = "Feature Importance",
        xlabel = "|Shapley effect| (baseline = $(tbl.baseline))",
        ylabel = "Feature",
        limits = (0, nothing, 0, nothing),
        yreversed = true,
    )
    barplot!(
        ax,
        tbl.x,
        tbl.y,
        direction = :x,
        # bar_labels = :y,
        # flip_labels_at = 0.3,
        # label_formatter = x-> "Flip at $(x)?",
    )
    if add_errorbars
        errorbars!(ax, tbl.y, tbl.x, tbl.y_sdom, whiskerwidth = 12, direction = :x)
    end

    if !savefig
        return f
    else
        save("$filename.pdf", f)
    end
end


function plot_shap_variable(data_shap; feature = "x1", xlim = nothing, ylim = nothing)

    data_plot = @chain data_shap begin
        @rsubset :feature_name == feature
    end
    # data_plot = data_shap[data_shap.feature_name .== "x1", :]

    baseline = round(data_shap.intercept[1], digits = 1)

    tbl = (
        x = int(data_plot.feature_value, type = Int64),
        y = data_plot.shap_effect,
        xticks = 1:4,
        name = levels(data_plot.feature_value),
        baseline = baseline,
    )

    f = Figure()
    ax = Axis(
        f[1, 1],
        xticks = (tbl.xticks, tbl.name),
        title = "Feature Importance for $feature",
        xlabel = feature,
        ylabel = "|Shapley effect| (baseline = $(tbl.baseline))",
    )
    violin!(ax, tbl.x, tbl.y, show_median = true)
    if xlim !== nothing
        xlims!(ax, xlim) # as vector
    end
    if ylim !== nothing
        ylims!(ax, ylim) # as vector
    end
    return f
end

function plot_shap_variables(
    data_shap;
    xlim = nothing,
    ylim = nothing,
    savefig = true,
    filename = "fig_shap_global",
)

    if xlim === nothing
        xlim = (0.5, 4.5)
    end
    if ylim === nothing
        ylim = (minimum(data_shap.shap_effect) - 0.1, maximum(data_shap.shap_effect) + 0.1)
    end

    features = unique(data_shap.feature_name)
    d_figures = OrderedDict{String,Figure}()

    for feature in features
        d_figures[feature] =
            plot_shap_variable(data_shap; feature = feature, xlim = xlim, ylim = ylim)
    end

    if !savefig
        return d_figures

    else

        files = String[]
        for (feature, fig) in d_figures
            file = "$(filename)__$feature.pdf"
            save(file, fig)
            push!(files, file)
        end
        run(`/usr/local/bin/pdftk $files cat output $filename.pdf`)

        # delete files
        for file in files
            rm(file)
        end

    end

end


#current_figure()