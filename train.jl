using Knet
using ArgParse
using JSON
using MAT
using JLD

include("util.jl")
include("model.jl")

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
        ("--generate"; action=:store_true)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    sr = o[:seed] > 0 ? srand(o[:seed]) : srand()

    # load data
    jsondata = JSON.parsefile(abspath(o[:jsonfile]))

    # TODO: data loading
    p2i, i2p = build_vocab(jsondata)
    @time trn = make_data(o[:features], jsondata, "train")
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
        w = map(wi->convert(o[:atype],wi), w)
    end
    opts = map(wi->eval(parse(o[:optim])), w)

    if o[:generate]
        tst = make_data(o[:features], jsondata, "test")
        tst = reduce(vcat, tst)
        @time lossval, acc = validate(w,val,p2i; o=o)
        println("lossval:$lossval,acc:$acc")
    end

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

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
