import MLJBase
using DataFrames
import DataFramesMeta
using BioSequences
using ShapML
using OrderedCollections
using ProgressMeter
using ColorSchemes
using GLMakie
using CairoMakie
using CategoricalArrays
using MLJ


d_base2int = Dict(
    # DNA_N => 0,
    DNA_A => 1,
    DNA_C => 2,
    DNA_G => 3,
    DNA_T => 4,
)
# d_int2base = Dict(i => base for (i, base) in d_base2int)
all_bases = sort(collect(keys(d_base2int)))
base_levels = Int8.(sort(collect(values(d_base2int))))

function convert_DNA_to_int(bases, datatype = Int8)
    bases_int = zeros(datatype, length(bases))
    for (i, base) in enumerate(bases)
        bases_int[i] = d_base2int[base]
    end
    return bases_int
end

function convert_DNAs_to_categorical(bases, datatype = Int8)
    bases_int = convert_DNA_to_int(bases)
    return CategoricalArray{datatype,1,UInt8}(bases_int, levels = base_levels)
end

function convert_DNA_dataframe_to_categorical(df)
    df = copy(df)
    for column in names(df)
        df[!, column] = convert_DNAs_to_categorical(df[!, column])
    end
    return df
end

# function bases2dataframe(bases, prefix)
#     ohe = all_bases .== permutedims(bases)
#     base_names = string.(Char.(all_bases))
#     column_names = [Symbol("$(prefix)_$(base)") for base in base_names]
#     return DataFrame(table(permutedims(ohe); names = column_names))
# end

# function dataframe2ohe(df)
#     hcat([bases2dataframe(df[!, column], column) for column in names(df)]...)
# end

function get_Xy(filename, N_rows = 1000)

    df = deserialize(filename).df
    if N_rows < size(df, 1)
        sample_rows = sample(1:nrow(df), N_rows, replace = false)
        df = df[sample_rows, :]
    end

    X = convert_DNA_dataframe_to_categorical(select(df, Not(:y)))
    y = CategoricalArray{Int8,1,Int8}(df.y)
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
    N_rows = Int(size(df, 1) // 4)
    m = permutedims(reshape(df.values, (4, N_rows)))
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

function print_performance(eval, model_name)
    μ = round(eval.measurement[2], sigdigits = 5)
    per_fold = eval.per_fold[2]
    σ = round(std(per_fold) / sqrt(length(per_fold)), digits = 5)
    println("$model_name = $μ ± $σ")
end;

#%%


function plot_LR_fit_coefficients(df)

    max_pos = size(df, 1)
    positions = 1:max_pos
    bases = [:A, :C, :G, :T]


    f = Figure()
    ax = Axis(
        f[1, 1],
        title = "LR fit coefficients as a function of position",
        xlabel = "Position",
        ylabel = "Fit coefficient (LR)",
        limits = (0.5, max_pos + 0.5, nothing, nothing),
        # xticks = 1:2:seq_length,
        )

    colormap = [x for x in ColorSchemes.Set1_9.colors]
    for (i, base) in enumerate(bases)
        scatterlines!(
            ax,
            positions,
            df[:, base],
            color = colormap[i],
            markercolor = colormap[i],
            label = String(base),
            )
    end

    axislegend(position = :cb, nbanks=4)
    return f
end



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
        save("./figures/$filename.pdf", f)
    end
end


function plot_shap_variable(data_shap; feature = "x1", xlim = nothing, ylim = nothing)

    data_plot = DataFramesMeta.@chain data_shap begin
        DataFramesMeta.@rsubset :feature_name == feature
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
            file = "./figures/$(filename)__$feature.pdf"
            save(file, fig)
            push!(files, file)
        end
        run(`/usr/local/bin/pdftk $files cat output ./figures/$filename.pdf`)

        # delete files
        for file in files
            rm(file)
        end

    end

end


#current_figure()

#%%

import Random.seed!;


function get_accuracies_pr_base(X)

    seq_length = Int((size(X, 2)) / 2)
    N_positions_vec = 1:seq_length

    accs_logreg = Float64[]
    accs_lgb_acc = Float64[]

    X_org = copy(X)

    p = Progress(sum(N_positions_vec), 1)   # minimum update interval: 1 second
    position = 0
    for N_positions in N_positions_vec

        ProgressMeter.update!(p, position)
        position += N_positions

        X = hcat(X_org[:, 1:N_positions], X_org[:, end-N_positions+1:end])

        seed!(42);
        mach_logreg = machine(pipe_logreg, X, y)
        fit!(mach_logreg, rows = train, verbosity = 0)
        acc_logreg = accuracy(predict_mode(mach_logreg, rows = test), y_test)
        push!(accs_logreg, acc_logreg)

        seed!(42);
        mach_lgb_cat = machine(pipe_lgb_cat, X, y)
        fit!(mach_lgb_cat, rows = train, verbosity = 0)
        acc_lgb_cat = accuracy(predict_mode(mach_lgb_cat, rows = test), y_test)
        push!(accs_lgb_acc, acc_lgb_cat)


    end

    accs = [
        ("N_positions_vec", N_positions_vec),
        ("Logistic Regression", accs_logreg),
        ("LightGBM (Cat)", accs_lgb_acc),
    ];

    return accs
end


function plot_accuracy_function_of_bases(accuracies; ylimits=(nothing, nothing))

    N_positions_vec = accuracies[1][2]
    seq_length = maximum(N_positions_vec)

    colormap = [x for x in ColorSchemes.Set1_9.colors]

    f = Figure()
    ax = Axis(
        f[1, 1],
        title = "Accuracy as a function of number of bases included (symmetric)",
        xlabel = "# bases included (symmetric)",
        ylabel = "Accuracy",
        # limits = (0.5, seq_length + 0.5, 0.634, 0.701),
        limits = (0.5, seq_length + 0.5, ylimits...),
        xticks = 1:2:seq_length,
        )



    for (i, acc) in enumerate(accuracies[2:end])
        scatterlines!(
            ax,
            N_positions_vec,
            acc[2],
            color = colormap[i],
            markercolor = colormap[i],
            label = acc[1],
            )
    end

    axislegend(position = :rb)
    return f
end

