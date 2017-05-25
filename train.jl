using Knet
using ArgParse
using JSON
using MAT
using JLD

include("util.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "CNN-HMM hybrid ASR system in Knet by Ilker Kesen."

    @add_arg_table s begin
        ("--features"; required=true; help="MFSC features file")
        ("--jsonfile"; required=true)
        ("--loadfile"; default=nothing)
        ("--savefile"; default=nothing)
        ("--hidden"; default=128; arg_type=Int64)
        ("--epochs"; default=100; arg_type=Int64)
        ("--batchsize"; default=200; arg_type=Int64)
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}"))
        ("--optim"; default="Adam(;gclip=5.0)")
        ("--pdrop"; default=0.0; arg_type=Float64)
        ("--seed"; default=-1; arg_type=Int64)
        ("--nvalid"; default=200; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()

    # load data
    jsondata = JSON.parsefile(abspath(o[:jsonfile]))

    # TODO: data loading
    p2i, i2p = build_vocab(jsondata)
    trn = make_data(o[:features], jsondata, "train")
    shuffle!(trn)
    val = []
    for k = 1:o[:nvalid]
        push!(val, pop!(trn))
    end
    trn = reduce(vcat, trn)
    val = reduce(vcat, val)

    # load model
    w = opts = bestacc = nothing
    if o[:loadfile] == nothing
        w = initweights(o[:atype], o[:hidden])
        bestacc = -Inf
    else
        w = load(o[:loadfile], "w")
        bestacc = load(o[:loadfile], "acc")
    end
    opts = map(wi->eval(parse(o[:optim])), w)

    bestacc = -Inf
    println("training has been started. ", now())
    for epoch = 1:o[:epochs]
        # shuffle train data
        shuffle!(trn)

        # one epoch training
        losstrn = 0
        @time for k = 1:o[:batchsize]:length(trn)
            samples = trn[k:min(length(trn),k+o[:batchsize]-1)]
            x = make_input(samples)
            ygold = make_output(samples,p2i)
            this_loss = train!(w,convert(o[:atype],x),ygold,opts;o=o)
            losstrn += this_loss
        end
        losstrn = losstrn / length(trn)

        # compute validation loss and accuracy
        @time lossval, acc = validate(w,val,p2i; o=o)

        # report and save
        print("(epoch:$epoch,losstrn:$losstrn,lossval:$lossval,acc:$acc) ")
        print(now())
        if acc > bestacc
            bestacc = acc
            save(o[:savefile], "w", map(wi->Array(wi),w), "acc", acc)
            print(" model saved!")
        end
        println()
    end
end

# TODO: implement this function
function initweights(atype, hidden, ysize=61)
    w = []
    push!(w, xavier(8,1,45,150))
    push!(w, zeros(1,1,150,1))
    push!(w, xavier(6,1,150,300))
    push!(w, zeros(1,1,300,1))
    push!(w, xavier(1000,1500))
    push!(w, zeros(1000,1))
    push!(w, xavier(1000,1000))
    push!(w, zeros(1000,1))
    push!(w, xavier(ysize,1000))
    push!(w,  zeros(ysize,1))
    return map(wi->convert(atype,wi), w)
end

function predict(w,x; o=Dict())
    # conv1
    x = conv4(w[1],x; padding=0) .+ w[2]
    x = relu(x)
    x = pool(x; window=(4,1), stride=(2,1))

    # conv2
    x = conv4(w[3],x; padding=0) .+ w[4]
    x = relu(x)
    x = pool(x; window=(2,1), stride=(2,1))


    # mlp
    x = relu(w[5] * mat(x) .+ w[6])
    x = dropout(x,get(o, :pdrop, 0.0))
    x = relu(w[7] * x .+ w[8])
    x = dropout(x,get(o, :pdrop, 0.0))

    # softmax
    x = w[end-1] * x .+ w[end]
end

function train!(w,x,y,opts; o=Dict())
    values = []
    g = lossgradient(w,x,y; o=o, values=values)
    update!(w,g,opts)
    lossval = values[1]
    return lossval
end

function loss(w, x, ygold; o=Dict(), values=[])
    ypred = predict(w, x; o=o)
    lossval = -logprob(ygold,ypred)
    push!(values, lossval)
    return lossval / length(ygold)
end

lossgradient = grad(loss)

function logprob(output, ypred)
    nrows,ncols = size(ypred)
    index = output + nrows*(0:(length(output)-1))
    o1 = logp(ypred,1)
    o2 = o1[index]
    o3 = sum(o2)
    return o3
end

function validate(w,val,p2i; o=Dict())
    batchsize = get(o, :batchsize, 100)
    lossval = losscnt = ncorrect = 0
    atype = get(o, :atype, gpu()>=0?KnetArray{Float32}:Array{Float32})
    for k = 1:batchsize:length(val)
        lower = k
        upper = min(length(val), k+batchsize-1)

        x = make_input(val[lower:upper])
        y = make_output(val[lower:upper], p2i)

        ypred = predict(w,convert(atype, x))
        lossval += -logprob(y,ypred)
        nrows,ncols = size(ypred)
        index = y + nrows*(0:(length(y)-1))
        ncorrect += sum(maximum(ypred,1).== ypred[index])
        losscnt += ncols
    end
    return lossval/losscnt, ncorrect/losscnt
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
