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
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()

    # load data
    jsondata = JSON.parsefile(abspath(o[:jsonfile]))
    filext = splitext(o[:features])[end]
    loadfun = filext == "mat" ? matread : load
    features = loadfun(o[:features])

    # build up vocabulary
    p2i, i2p = build_vocab(jsondata)

    # TODO: data loading
    trn = val = nothing

    # load model
    w = opts = bestacc = nothing
    if o[:loafile] == nothing
        w = initweights(o[:atype], o[:hidden])
        bestacc = -Inf
    else
        w = load(o[:loadfile], "w")
        bestacc = load(o[:loadfile], "acc")
    end
    opts = map(wi->eval(parse(o[:optim])), w)

    bestacc = -Inf
    for epoch = 1:o[:epochs]
        # shuffle train data
        shuffle!(trn)

        # one epoch training
        losstrn = 0
        for k = 1:o[:batchsize]:length(trn)
            samples = trn[k:min(length(trn),k+batchsize-1)]
            x = make_input(samples)
            ygold = make_output(samples)
            this_loss = train!(w,x,ygold,opts;o=o)
            losstrn += this_loss
        end
        losstrn = losstrn / length(trn)

        # compute validation loss and accuracy
        lossval, acc = validate(w,val)

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

function initweights(atype, hidden, xsize, ysize)

end

function predict(w,x; o=Dict())
    for i = 1:2:length(w)-4
        x = conv4(w[i],x) .+ w[i+1]
        x = relu(x)
        x = pool(x)
    end

    x = relu(w[end-3] * mat(x) .+ w[end-2])
    x = dropout(x,get(o, :pdrop, 0.0))
    x = w[end-1] * x .+ w[end]
end

function train!(w,x,y,opts; o=Dict())
    values = []
    g = lossgradient(w,x,y; o=o, values=values)
    update!(w,g,opt)
    lossval = values[1]
    return lossval
end

function loss(w, x, ygold; o=Dict(), values=[])
    ypred = predict(w, x)
    lossval = -logprob(ygold,ypred)
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

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
