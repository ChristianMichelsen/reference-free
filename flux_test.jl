using StableRNGs


include("mlj_functions.jl")

#%%

filename = "./data/df.data"
# N_rows = 1_000_000
N_rows = 10_000

X, y = get_Xy(filename, N_rows);
train, test = partition(eachindex(y), 0.75; shuffle = true, rng = StableRNG(123));

X_train = selectrows(X, train);
y_train = selectrows(y, train);
X_test = selectrows(X, test);
y_test = selectrows(y, test);

hot = OneHotEncoder();
mach = fit!(machine(hot, X))
Xt = MLJ.transform(mach, X)
Xt |> schema


using MLJ
using Flux
using MLDatasets

# helper function
function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

import MLJFlux
mutable struct MyConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
end

function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)

    k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3

    mod(k, 2) == 1 || error("`filter_size` must be odd. ")

    # padding to preserve image size on convolution:
    p = div(k - 1, 2)

    front = Chain(
        Conv((k, k), n_channels => c1, pad = (p, p), relu),
        MaxPool((2, 2)),
        Conv((k, k), c1 => c2, pad = (p, p), relu),
        MaxPool((2, 2)),
        Conv((k, k), c2 => c3, pad = (p, p), relu),
        MaxPool((2, 2)),
        flatten,
    )
    d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
    return Chain(front, Dense(d, n_out))
end


N = 500
Xraw, yraw = MNIST.traindata();
Xraw = Xraw[:, :, 1:N];
yraw = yraw[1:N];

scitype(Xraw)
scitype(yraw)


X = coerce(Xraw, GrayImage);
y = coerce(yraw, Multiclass);

ImageClassifier = @load ImageClassifier
clf = ImageClassifier(
    builder = MyConvBuilder(3, 16, 32, 32),
    epochs = 10,
    loss = Flux.crossentropy,
)

machine(clf, X, y)

evaluate!(
    mach,
    resampling = Holdout(rng = 123, fraction_train = 0.7),
    operation = predict_mode,
    measure = misclassification_rate,
)
