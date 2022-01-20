
using MLJ
using Flux
import MLJFlux
import MLJIteration # for `skip`
using CairoMakie
using Flux: onehotbatch

# ## Basic training

# Downloading the MNIST image dataset:

# import MLDatasets: MNIST
# images, labels = MNIST.traindata();

# mask = [label in [0, 1] for label in labels];
# images = images[:, :, mask];
# labels = labels[mask];


# # map(image -> image[14:17, :], images)
# # [image[14:17, :] for image in images]


# # In MLJ, integers cannot be used for encoding categorical data, so we
# # must force the labels to have the `Multiclass` [scientific
# # type](https://juliaai.github.io/ScientificTypes.jl/dev/). For
# # more on this, see [Working with Categorical
# # Data](https://alan-turing-institute.github.io/MLJ.jl/dev/working_with_categorical_data/).
# labels = coerce(labels, Binary);
# images = coerce(images, GrayImage);


# # Checking scientific types:
# @assert scitype(images) <: AbstractVector{<:MLJ.Image}
# @assert scitype(labels) <: AbstractVector{<:Finite}


# # Looks good.

# # For general instructions on coercing image data, see [Type coercion
# # for image
# # data](https://alan-turing-institute.github.io/ScientificTypes.jl/dev/#Type-coercion-for-image-data-1)

# images[1]
# # images[1][14:17, :]

# # images = [image[:, 14:17] for image in images]
# # images = [image[14:17, :] for image in images]



# # We start by defining a suitable `Builder` object. This is a recipe
# # for building the neural network. Our builder will work for images of
# # any (constant) size, whether they be color or black and white (ie,
# # single or multi-channel).  The architecture always consists of six
# # alternating convolution and max-pool layers, and a final dense
# # layer; the filter size and the number of channels after each
# # convolution layer is customisable.

# import MLJFlux
struct MyConvBuilder
    filter_size::Int
    channels1::Int
    channels2::Int
    channels3::Int
end

# make2d(x::AbstractArray) = reshape(x, :, size(x)[end])

# function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)
#     k, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3
#     mod(k, 2) == 1 || error("`filter_size` must be odd. ")
#     p = div(k - 1, 2) # padding to preserve image size
#     init = Flux.glorot_uniform(rng)
#     front = Chain(
#         Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
#         MaxPool((2, 2)),
#         Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
#         MaxPool((2, 2)),
#         Conv((k, k), c2 => c3, pad = (p, p), relu, init = init),
#         MaxPool((2, 2)),
#         make2d,
#     )
#     d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
#     return Chain(front, Dense(d, n_out, init = init))
# end



# # **Note.** There is no final `softmax` here, as this is applied by
# # default in all MLJFLux classifiers. Customisation of this behaviour
# # is controlled using using the `finaliser` hyperparameter of the
# # classifier.

# # We now define the MLJ model. If you have a GPU, substitute
# # `acceleration=CUDALibs()` below:

# ImageClassifier = @load ImageClassifier
# clf = ImageClassifier(
#     builder = MyConvBuilder(3, 16, 32, 32),
#     batch_size = 50,
#     epochs = 10,
#     rng = 123,
# )


# # You can add Flux options `optimiser=...` and `loss=...` here. At
# # present, `loss` must be a Flux-compatible loss, not an MLJ
# # measure. To run on a GPU, set `acceleration=CUDALib()`.

# # Binding the model with data in an MLJ machine:
# mach = machine(clf, images, labels);


# # Training for 10 epochs on the first 500 images:
# fit!(mach, rows = 1:500, verbosity = 2);


# # Inspecting:
# report(mach)

# chain = fitted_params(mach)

# Flux.params(chain)[2]


# # Adding 20 more epochs:
# clf.epochs = clf.epochs + 20
# fit!(mach, rows = 1:500);


# # Computing an out-of-sample estimate of the loss:
# predicted_labels = predict(mach, rows = 501:1000);
# cross_entropy(predicted_labels, labels[501:1000]) |> mean


# # Or, in one line:
# evaluate!(
#     mach,
#     resampling = Holdout(fraction_train = 0.5),
#     measure = cross_entropy,
#     rows = 1:1000,
#     verbosity = 0,
# )



# # ## Wrapping the MLJFlux model with iteration controls

# # Any iterative MLJFlux model can be wrapped in *iteration controls*,
# # as we demonstrate next. For more on MLJ's `IteratedModel` wrapper,
# # see the [MLJ
# # documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/).

# # The "self-iterating" classifier, called `iterated_clf` below, is for
# # iterating the image classifier defined above until one of the
# # following stopping criterion apply:

# # - `Patience(3)`: 3 consecutive increases in the loss
# # - `InvalidValue()`: an out-of-sample loss, or a training loss, is `NaN`, `Inf`, or `-Inf`
# # - `TimeLimit(t=5/60)`: training time has exceeded 5 minutes

# # These checks (and other controls) will be applied every two epochs
# # (because of the `Step(2)` control). Additionally, training a
# # machine bound to `iterated_clf` will:

# # - save a snapshot of the machine every three control cycles (every six epochs)
# # - record traces of the out-of-sample loss and training losses for plotting
# # - record mean value traces of each Flux parameter for plotting

# # For a complete list of controls, see [this
# # table](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided).

# # ### Wrapping the classifier

# # Some helpers

# make1d(x::AbstractArray) = reshape(x, length(x));


# # To extract Flux params from an MLJFlux machine
# parameters(mach) = make1d.(Flux.params(fitted_params(mach)));


# # To store the traces:
# losses = Float32[]
# training_losses = Float32[]
# parameter_means = Float32[];
# epochs = Int64[]


# # To update the traces:
# update_loss(loss) = push!(losses, loss)
# update_training_loss(losses) = push!(training_losses, losses[end])
# update_means(mach) = append!(parameter_means, mean.(parameters(mach)));
# update_epochs(epoch) = push!(epochs, epoch)


# # The controls to apply:
# save_control = MLJIteration.skip(Save("mnist.jlso"), predicate = 3)
# controls = [
#     Step(2),
#     Patience(3),
#     InvalidValue(),
#     TimeLimit(5 / 60),
#     NumberLimit(n = 10),
#     save_control,
#     WithLossDo(),
#     WithLossDo(update_loss),
#     WithTrainingLossesDo(update_training_loss),
#     Callback(update_means),
#     WithIterationsDo(update_epochs),
# ];


# # The "self-iterating" classifier:
# iterated_clf = IteratedModel(
#     model = clf,
#     controls = controls,
#     resampling = Holdout(fraction_train = 0.7),
#     measure = log_loss,
# )


# # ### Binding the wrapped model to data:
# mach = machine(iterated_clf, images, labels);


# # ### Training
# fit!(mach, rows = 1:500);


# #%%

# ### Comparison of the training and out-of-sample losses:
# f = Figure()
# ax = Axis(f[1, 1], xlabel = "epoch", ylabel = "root squared error")
# scatter!(ax, epochs, losses, label = "out-of-sample")
# scatter!(ax, epochs, training_losses, label = "training")
# axislegend(ax, position = :rb)
# f


# # ### Evolution of weights
# n_epochs = length(losses)
# n_parameters = div(length(parameter_means), n_epochs)
# parameter_means2 = reshape(copy(parameter_means), n_parameters, n_epochs)'
# f = Figure()
# ax = Axis(f[1, 1], xlabel = "epoch", ylabel = "root squared error")
# for i = 1:size(parameter_means2, 2)
#     scatter!(ax, epochs, parameter_means2[:, i], label = "y$i")
# end
# axislegend(ax, position = :rb)
# f

# # ### Retrieving a snapshot for a prediction:
# mach2 = machine("mnist3.jlso")
# predict_mode(mach2, images[501:503])



# # ### Restarting training

# # Mutating `iterated_clf.controls` or `clf.epochs` (which is otherwise
# # ignored) will allow you to restart training from where it left off.

# iterated_clf.controls[2] = Patience(4)
# fit!(mach, rows = 1:500)

# #%%

# # layer = Conv((4, ), 1=>1)

# # x1 = Float32[[1,2,3,4];;;]
# # x2 = Float32[[1,2,3,4];;; [2,3,4,5]]

# # layer(x1)
# # layer(x2)





# # xtrain, ytrain = MNIST.traindata(Float32)
# # xtest, ytest = MNIST.testdata(Float32)

# # xtrain = reshape(xtrain, 28, 28, 1, :)
# # xtest = reshape(xtest, 28, 28, 1, :)

# # using Flux: onehotbatch
# # onehotbatch(ytrain, 0:9)
# # size(xtrain)


# #%%

# # p = div(k - 1, 2) # padding to preserve image size
# #     init = Flux.glorot_uniform(rng)
# #     front = Chain(
# #         Conv((k, k), n_channels => c1, pad = (p, p), relu, init = init),
# #         MaxPool((2, 2)),
# #         Conv((k, k), c1 => c2, pad = (p, p), relu, init = init),
# #         MaxPool((2, 2)),
# #         Conv((k, k), c2 => c3, pad = (p, p), relu, init = init),
# #         MaxPool((2, 2)),
# #         make2d,
# #     )


# # image_shape = (28, 28, 1, 1)
# # image = rand(Float32, image_shape...);

# # k = 3
# # p = div(k - 1, 2) # padding to preserve image size
# # c1 = 16
# # c2 = 32
# # c3 = 32

# # init = Flux.glorot_uniform()

# # layer1 = Conv((k, k), 1 => c1, pad = (p, p), relu, init = init)
# # layer1(image) |> size
# # layer2 = MaxPool((2, 2))
# # layer1(image) |> layer2 |> size
# # layer3 = Conv((k, k), c1 => c2, pad = (p, p), relu, init = init)
# # layer1(image) |> layer2 |> layer3 |> size
# # layer4 = MaxPool((2, 2))
# # layer1(image) |> layer2 |> layer3 |> layer4 |> size
# # layer5 = Conv((k, k), c2 => c3, pad = (p, p), relu, init = init)
# # layer1(image) |> layer2 |> layer3 |> layer4 |> layer5 |> size
# # layer6 = MaxPool((2, 2))
# # layer1(image) |> layer2 |> layer3 |> layer4 |> layer5 |> layer6 |> size
# # layer1(image) |> layer2 |> layer3 |> layer4 |> layer5 |> layer6 |> make2d |> size


# # # make2d(x::AbstractArray) = reshape(x, :, size(x)[end])



# # #%%

# # image_shape = (76, 4, 1, 10)
# # image = rand(Float32, image_shape...);

# # kernel_size = 5
# # filters = 32

# # layer1 = Conv((kernel_size, 4), 1=>filters, relu, pad = (2, 0))
# # layer1(image) |> size

# # layer2 = Conv((kernel_size, 1), filters=>filters, relu, pad = (2, 0))
# # layer1(image) |> layer2 |> size

# # layer3 = MaxPool((4, 1))
# # layer1(image) |> layer2 |> layer3 |> size

# # layer1(image) |> layer2 |> layer3 |> make2d |> size


#%%

using Serialization
using StatsBase: sample
using BioSequences
using DataFrames


filename = "./data/df.data"
df = deserialize(filename).df
N_rows = 10000
sample_rows = sample(1:nrow(df), N_rows, replace = false)
df = df[sample_rows, :]

X = select(df, Not(:y))
y = df.y

function onehotrow(X_row, alphabet)
    return permutedims(onehotbatch(X_row, alphabet))
end

function onehotrow(X_row)
    alphabet = [DNA_A, DNA_C, DNA_G, DNA_T]
    return permutedims(onehotbatch(X_row, alphabet))
end


images = cat([onehotrow(values(X[i, :])) for i = 1:N_rows]..., dims = 3)

make2d(x::AbstractArray) = reshape(x, :, size(x)[end])


function MLJFlux.build(b::MyConvBuilder, rng, n_in, n_out, n_channels)
    kernel_size, c1, c2, c3 = b.filter_size, b.channels1, b.channels2, b.channels3
    mod(kernel_size, 2) == 1 || error("`filter_size` must be odd. ")
    # p = div(k - 1, 2) # padding to preserve image size
    init = Flux.glorot_uniform(rng)
    front = Chain(
        Conv((kernel_size, 4), 1 => c1, relu, pad = (2, 0), init = init),
        # MaxPool((2, 2)),
        Conv((kernel_size, 1), c1 => c2, pad = (2, 0), relu, init = init),
        MaxPool((2, 1)),
        # Conv((k, k), c2 => c3, pad = (p, p), relu, init = init),
        # MaxPool((2, 2)),
        make2d,
    )
    d = Flux.outputsize(front, (n_in..., n_channels, 1)) |> first
    return Chain(front, Dense(d, n_out, init = init))
end


ImageClassifier = @load ImageClassifier
clf = ImageClassifier(
    builder = MyConvBuilder(5, 32, 32, 0),
    batch_size = 50,
    epochs = 10,
    rng = 123,
)


labels = y


# images = images[:, 10:13, :];
size(images)

images = coerce(images, GrayImage);
labels = coerce(labels, Binary);

# Checking scientific types:
@assert scitype(images) <: AbstractVector{<:MLJ.Image}
@assert scitype(labels) <: AbstractVector{<:Finite}


images[1]'


ImageClassifier = @load ImageClassifier
clf = ImageClassifier(
    builder = MyConvBuilder(3, 16, 32, 32),
    batch_size = 50,
    epochs = 10,
    rng = 123,
)


# You can add Flux options `optimiser=...` and `loss=...` here. At
# present, `loss` must be a Flux-compatible loss, not an MLJ
# measure. To run on a GPU, set `acceleration=CUDALib()`.

# Binding the model with data in an MLJ machine:
mach = machine(clf, images, labels);


# Training for 10 epochs on the first 500 images:
fit!(mach, rows = 1:500, verbosity = 2);


# Inspecting:
report(mach)

chain = fitted_params(mach)

Flux.params(chain)[2]


# Adding 20 more epochs:
clf.epochs = clf.epochs + 20
fit!(mach, rows = 1:500);


# Computing an out-of-sample estimate of the loss:
predicted_labels = predict(mach, rows = 501:1000);
cross_entropy(predicted_labels, labels[501:1000]) |> mean


# Or, in one line:
evaluate!(
    mach,
    resampling = Holdout(fraction_train = 0.5),
    measure = cross_entropy,
    rows = 1:1000,
    verbosity = 0,
)



# ## Wrapping the MLJFlux model with iteration controls

# Any iterative MLJFlux model can be wrapped in *iteration controls*,
# as we demonstrate next. For more on MLJ's `IteratedModel` wrapper,
# see the [MLJ
# documentation](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/).

# The "self-iterating" classifier, called `iterated_clf` below, is for
# iterating the image classifier defined above until one of the
# following stopping criterion apply:

# - `Patience(3)`: 3 consecutive increases in the loss
# - `InvalidValue()`: an out-of-sample loss, or a training loss, is `NaN`, `Inf`, or `-Inf`
# - `TimeLimit(t=5/60)`: training time has exceeded 5 minutes

# These checks (and other controls) will be applied every two epochs
# (because of the `Step(2)` control). Additionally, training a
# machine bound to `iterated_clf` will:

# - save a snapshot of the machine every three control cycles (every six epochs)
# - record traces of the out-of-sample loss and training losses for plotting
# - record mean value traces of each Flux parameter for plotting

# For a complete list of controls, see [this
# table](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided).

# ### Wrapping the classifier

# Some helpers

make1d(x::AbstractArray) = reshape(x, length(x));


# To extract Flux params from an MLJFlux machine
parameters(mach) = make1d.(Flux.params(fitted_params(mach)));


# To store the traces:
losses = Float32[]
training_losses = Float32[]
parameter_means = Float32[];
epochs = Int64[]


# To update the traces:
update_loss(loss) = push!(losses, loss)
update_training_loss(losses) = push!(training_losses, losses[end])
update_means(mach) = append!(parameter_means, mean.(parameters(mach)));
update_epochs(epoch) = push!(epochs, epoch)


# The controls to apply:
save_control = MLJIteration.skip(Save("mnist.jlso"), predicate = 3)
controls = [
    Step(2),
    Patience(3),
    InvalidValue(),
    TimeLimit(5 / 60),
    NumberLimit(n = 100),
    save_control,
    WithLossDo(),
    WithLossDo(update_loss),
    WithTrainingLossesDo(update_training_loss),
    Callback(update_means),
    WithIterationsDo(update_epochs),
];


# The "self-iterating" classifier:
iterated_clf = IteratedModel(
    model = clf,
    controls = controls,
    resampling = Holdout(fraction_train = 0.7),
    measure = log_loss,
)


# ### Binding the wrapped model to data:
mach = machine(iterated_clf, images, labels);


# ### Training
fit!(mach, rows = 1:500);


#%%

### Comparison of the training and out-of-sample losses:
f = Figure()
ax = Axis(f[1, 1], xlabel = "epoch", ylabel = "root squared error")
scatter!(ax, epochs, losses, label = "out-of-sample")
scatter!(ax, epochs, training_losses, label = "training")
axislegend(ax, position = :rb)
f


# ### Evolution of weights
n_epochs = length(losses)
n_parameters = div(length(parameter_means), n_epochs)
parameter_means2 = reshape(copy(parameter_means), n_parameters, n_epochs)'
f = Figure()
ax = Axis(f[1, 1], xlabel = "epoch", ylabel = "root squared error")
for i = 1:size(parameter_means2, 2)
    scatter!(ax, epochs, parameter_means2[:, i], label = "y$i")
end
axislegend(ax, position = :rb)
f

# ### Retrieving a snapshot for a prediction:
mach2 = machine("mnist3.jlso")
predict_mode(mach2, images[501:503])



# ### Restarting training

# Mutating `iterated_clf.controls` or `clf.epochs` (which is otherwise
# ignored) will allow you to restart training from where it left off.

iterated_clf.controls[2] = Patience(4)
fit!(mach, rows = 1:500)
