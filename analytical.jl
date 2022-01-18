using Distributions
using StatsBase

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
    p_α_sig = 1 - p_GC_sig
    p_α_bkg = 1 - p_GC_bkg
    return mean([Binom_sum(N, p_α_sig), Binom_sum(N, p_α_bkg)], weights(weight))
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

using CairoMakie
using ColorSchemes


colormap = [x for x in ColorSchemes.Set1_9.colors]

#%%

f = Figure()
ax = Axis(
    f[1, 1],
    title = "Analytical accuracy",
    xlabel = "# bases included (centered)",
    ylabel = "Accuracy",
    # limits = (0.5, half_seq_length + 0.5, 0.634, 0.701),
    # limits = (0.5, half_seq_length + 0.5, ylimits...),
    xticks = 1:2:half_seq_length,
)

scatterlines!(
    ax,
    bases_included,
    accuracies,
    color = colormap[3],
    markercolor = colormap[3],
    label = "Analytical",
)

axislegend(position = :rb)
f


weights