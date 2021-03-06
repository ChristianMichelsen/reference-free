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
using NamedArrays
using Statistics
using Serialization
using Distributions
using StatsBase: weights
using StatsBase: sample
using StatsBase: countmap



d_base2int = Dict(
    # DNA_N => 0,
    DNA_A => 1,
    DNA_C => 2,
    DNA_G => 3,
    DNA_T => 4,
)
d_int2base = Dict(i => base for (base, i) in d_base2int)
d_int2string = Dict(i => string(Char(base)) for (base, i) in d_base2int)
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
    if 0 < N_rows && N_rows < size(df, 1)
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

function get_df_LR_long(mach::Machine)
    params_raw = fitted_params(mach)
    params = hcat(
        # DataFrame(intercept = params_raw.logistic_classifier.intercept),
        DataFrame(params_raw.logistic_classifier.coefs),
    )
    params_T = rename(transpose(params), :2 => :values)
    return params_T
end


function get_df_LR_wide(df::DataFrame)
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

function print_performance(eval, model_name, multiplier = 100, sigdigits = 3)
    ?? = round(multiplier * eval.measurement[2], sigdigits = sigdigits)
    per_fold = eval.per_fold[2]
    ?? = round(multiplier * std(per_fold) / sqrt(length(per_fold)), digits = sigdigits)
    println("$model_name = ($?? ?? $??) %")
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

    axislegend(position = :cb, nbanks = 4)
    return f
end



#%%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                     ??????   ??????  ??????????????????  ??????????????????
#                      ?????? ??????  ??????       ??????   ??????
#                       ?????????   ??????   ????????? ??????????????????
#                      ?????? ??????  ??????    ?????? ??????   ??????
#                     ??????   ??????  ??????????????????  ??????????????????
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


function MLJBase.accuracy(yhat::MLJBase.UnivariateFiniteArray, y::MLJ.CategoricalVector)
    return accuracy(mode.(yhat), y)
end

function MLJBase.confusion_matrix(
    yhat::MLJBase.UnivariateFiniteArray,
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
        auc_ = round(100 * area_under_curve(y_hat, y), digits = 1)
        legend_name = "$(name), AUC: $(auc_)%"
        lines!(ax, fprs, tprs, label = legend_name)
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
        name = string.(levels(data_plot.feature_value)),
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


function get_accuracies_pr_base(X, y, train, test)

    half_seq_length = Int((size(X, 2)) / 2)
    N_positions_vec = 1:half_seq_length

    accs_LR = Float64[]
    accs_LGB = Float64[]

    X_org = copy(X)

    p = Progress(sum(N_positions_vec), 1)   # minimum update interval: 1 second
    position = 0
    for N_positions in N_positions_vec

        ProgressMeter.update!(p, position)
        position += N_positions

        X_cut = hcat(X_org[:, 1:N_positions], X_org[:, end-N_positions+1:end])

        seed!(42)
        mach_LR = machine(pipe_LR, selectrows(X_cut, train), selectrows(y, train))
        fit!(mach_LR, verbosity = 0)
        acc_LR = accuracy(predict(mach_LR, selectrows(X_cut, test)), selectrows(y, test))
        push!(accs_LR, acc_LR)

        seed!(42)
        mach_LGB = machine(pipe_LGB, selectrows(X_cut, train), selectrows(y, train))
        fit!(mach_LGB, verbosity = 0)
        acc_LGB = accuracy(predict(mach_LGB, selectrows(X_cut, test)), selectrows(y, test))
        push!(accs_LGB, acc_LGB)

    end

    accs = [
        ("N_positions_vec", N_positions_vec),
        ("Logistic Regression", accs_LR),
        ("LightGBM", accs_LGB),
    ]

    return accs
end


function add_missing_bases!(d)
    k = keys(d)
    for i in 1:4
        if !(i in k)
            d[i] = 0
        end
    end
end


function predict_custom(X_row::DataFrameRow)
    d = countmap(X_row)
    add_missing_bases!(d)

    AT = d[1]+d[4]
    GC = d[3]+d[2]
    if AT > GC
        return 1
    elseif AT == GC
        return rand([0, 1])
    else
        return 0
    end
end

function predict_custom(X::DataFrame, N=-1)
    if N < 0
        N = size(X, 1)
    end
    return [predict_custom(X[i, :]) for i in 1:N]
end



function get_accuracies_pr_base_centered(X, y, train, test; add_analytical = true, add_custom=true)


    half_seq_length = Int((size(X, 2)) / 2)
    N_positions_vec = 0:half_seq_length-1

    accs_LR = Float64[]
    accs_LGB = Float64[]
    accs_custom = Float64[]

    X_org = copy(X)

    p = Progress(sum(N_positions_vec), 1)   # minimum update interval: 1 second
    position = 0
    for N_positions in N_positions_vec

        ProgressMeter.update!(p, position)
        position += N_positions

        middle_idxs = half_seq_length-N_positions:half_seq_length+N_positions+1
        middle_X = X_org[:, middle_idxs]

        seed!(42)
        mach_LR = machine(pipe_LR, selectrows(middle_X, train), selectrows(y, train))
        fit!(mach_LR, verbosity = 0)
        acc_LR = accuracy(predict(mach_LR, selectrows(middle_X, test)), selectrows(y, test))
        push!(accs_LR, acc_LR)

        seed!(42)
        mach_LGB = machine(pipe_LGB, selectrows(middle_X, train), selectrows(y, train))
        fit!(mach_LGB, verbosity = 0)
        acc_LGB =
            accuracy(predict(mach_LGB, selectrows(middle_X, test)), selectrows(y, test))
        push!(accs_LGB, acc_LGB)

        if add_custom
            seed!(42)
            yhat_custom = predict_custom(selectrows(middle_X, test));
            acc_custom = accuracy(yhat_custom, selectrows(y, test))
            push!(accs_custom, acc_custom)
        end

    end


    accs = [
        ("N_positions_vec", N_positions_vec),
        ("Logistic Regression", accs_LR),
        ("LightGBM", accs_LGB),
    ]

    if add_analytical
        analytical_accuracies = compute_analytical_accuracies(
            N_half = half_seq_length,
            p_GC_sig = 0.4,
            p_GC_bkg = 0.5,
            weight = [1, 1],
        )
        push!(accs, ("Analytical", analytical_accuracies))
    end

    if add_custom
        push!(accs, ("Custom", accs_custom))
    end

    return accs
end




function plot_accuracy_function_of_bases(accuracies; ylimits = (nothing, nothing))

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




function plot_accuracy_function_of_bases_centered(accuracies; ylimits = (nothing, nothing))

    N_positions_vec = accuracies[1][2] .+ 1
    half_seq_length = maximum(N_positions_vec)

    colormap = [x for x in ColorSchemes.Set1_9.colors]

    f = Figure()
    ax = Axis(
        f[1, 1],
        title = "Accuracy as a function of number of bases included (centered)",
        xlabel = "# bases included (centered)",
        ylabel = "Accuracy",
        # limits = (0.5, half_seq_length + 0.5, 0.634, 0.701),
        limits = (0.5, half_seq_length + 0.5, ylimits...),
        xticks = 1:2:half_seq_length,
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





#%%


function get_scores(yhat, label = 1)
    return pdf.(yhat, Int8(label))
end

function plot_score_density(yhat, y_test, title = "")

    yscores = get_scores(yhat)

    mask_signal = (y_test .== 1)
    mask_background = .!mask_signal
    yscores_signal = yscores[mask_signal]
    yscores_background = yscores[mask_background]

    f = Figure()
    ax = Axis(
        f[1, 1],
        title = title,
        xlabel = "Score",
        ylabel = "Density",
        limits = (0, 1, 0, nothing),
    )

    density!(
        ax,
        yscores_signal,
        color = (:red, 0.1),
        strokecolor = :red,
        strokewidth = 3,
        strokearound = false,
        label = "Signal",
    )
    density!(
        ax,
        yscores_background,
        color = (:blue, 0.1),
        strokecolor = :blue,
        strokewidth = 3,
        strokearound = false,
        label = "Background",
    )
    axislegend(position = :lt, nbanks = 4)

    return f
end


#%%


function tostring(base::DNA)
    return string(Char(base))
end

function get_firstbase_lastbase_mask(X, firstbase, lastbase)
    return (X[:, 1] .== d_base2int[firstbase]) .&& (X[:, end] .== d_base2int[lastbase])
end


function get_base_stratified_measure(X, yhat, y_test, measure_func = area_under_curve)

    # acc_table["A", "C"]
    # acc_table["Base 1" => "A", "Base 76" => "C"]

    base_stratified_measure = Float64[]
    for firstbase in all_bases
        for lastbase in all_bases
            mask_first_last = get_firstbase_lastbase_mask(X, firstbase, lastbase)
            # acc = measure(mode.(yhat[mask_first_last]), y_test[mask_first_last])
            measure = measure_func(yhat[mask_first_last], y_test[mask_first_last])
            push!(base_stratified_measure, measure)
        end
    end

    base_stratified_measure_2d = permutedims(reshape(base_stratified_measure, (4, 4)))
    measure_table =
        100 .* NamedArray(
            base_stratified_measure_2d,
            (tostring.(all_bases), tostring.(all_bases)),
            ("Base 1", "Base 76"),
        )
    return measure_table
end



function Statistics.mean(x::CategoricalVector)
    return mean(int(x) .- 1)
end

function Base.sum(x::CategoricalVector)
    return sum(int(x) .- 1)
end

function get_base_stratified_signal_proportion(X, y_test, func = mean, datatype = Float64)
    base_stratified_means = datatype[]
    for firstbase in all_bases
        for lastbase in all_bases
            mask_first_last = get_firstbase_lastbase_mask(X, firstbase, lastbase)
            push!(base_stratified_means, func(y_test[mask_first_last]))
        end
    end
    base_stratified_means_2d = permutedims(reshape(base_stratified_means, (4, 4)))
    acc_table =
        100 .* NamedArray(
            base_stratified_means_2d,
            (tostring.(all_bases), tostring.(all_bases)),
            ("Base 1", "Base 76"),
        )
    return acc_table
end


#%%


function get_base_counts_pr_position(X; normalise = true)
    df = vcat(
        [DataFrame(Dict("$i" => sum(X[:, col] .== i) for i = 1:4)) for col in names(X)]...,
    )
    rename!(df, d_int2string)
    if normalise
        df = df ./ size(X, 1)
    end

    return df
end

function make_base_fraction_plot(base_counts_signal, base_counts_background)

    N = size(base_counts_signal, 1)

    f = Figure()
    ax = Axis(
        f[1, 1],
        title = "Base fraction for signal (ancient) and background (modern)",
        xlabel = "Read position",
        ylabel = "Base fraction",
        limits = (0.5, N + 0.5, 0.1, 0.4),
        xticks = [1; collect(5:5:N)],
    )

    colormap = [color for color in ColorSchemes.Set1_9.colors]

    for (i, column) in enumerate(names(base_counts_signal))
        lines!(
            ax,
            1:N,
            base_counts_signal[:, column],
            color = colormap[i],
            markercolor = colormap[i],
            label = "Signal, Base: $column",
        )
    end

    for (i, column) in enumerate(names(base_counts_background))
        lines!(
            ax,
            1:N,
            base_counts_background[:, column],
            color = colormap[i],
            markercolor = colormap[i],
            label = "Background, Base: $column",
            linestyle = :dash,
        )
    end

    axislegend(position = :cb, nbanks = 4, orientation = :horizontal)
    return f
end

#%%

function Binom(k, N, p)
    return pdf(Binomial(N, p), k)
end

function Binom_sum(N, p)
    if iseven(N)
        N -= 1
    end
    return sum([Binom(k, N, p) for k = N:-1:N/2])
end


function compute_analytical_accuracy(; N, p_GC_sig, p_GC_bkg, weight = [1, 1])
    p_??_sig = 1 - p_GC_sig
    p_??_bkg = 1 - p_GC_bkg
    return mean([Binom_sum(N, p_??_sig), Binom_sum(N, p_??_bkg)], weights(weight))
end

function compute_analytical_accuracies(;
    N_half,
    p_GC_sig = 0.4,
    p_GC_bkg = 0.5,
    weight = [1, 1],
)
    bases_included = 1:N_half
    analytical_accuracies = [
        compute_analytical_accuracy(
            N = i * 2,
            p_GC_sig = p_GC_sig,
            p_GC_bkg = p_GC_bkg,
            weight = weight,
        ) for i in bases_included
    ]
    return analytical_accuracies
end
