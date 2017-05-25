using Knet
using ArgParse
using JLD,MAT,JSON,HDF5

include("util.jl")
include("model.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "CNN-HMM hybrid ASR system in Knet by Ilker Kesen."

    @add_arg_table s begin
        ("--features"; required=true; help="MFSC features file")
        ("--jsonfile"; required=true)
        ("--model"; default=nothing)
        ("--probs"; default=nothing)
        ("--savefile"; default=nothing)
        ("--batchsize"; default=400; arg_type=Int64)
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}"))
        ("--period"; default=100; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    atype = o[:atype]

    # load data
    jsondata = JSON.parsefile(o[:jsonfile])
    p2i, i2p = build_vocab(jsondata)

    # make data
    trnids, tstids = [], []
    trn = make_data(o[:features], jsondata; extra=trnids)
    tst = make_data(o[:features], jsondata, "test"; extra=tstids)

    # load model
    posteriors = load(o[:probs], "posteriors")
    priors = load(o[:probs], "priors")
    w = load(o[:model], "w")

    # convert atype
    # posteriors = convert(atype, posteriors)
    priors = convert(atype, priors)
    w = map(wi->convert(atype,wi), w)

    solutions = Dict()
    @time for (split,data) in zip(("train","test"),(trn,tst))
        getid(i) = split=="train"?trnids[i]:tstids[i]
        for (i,sample) in enumerate(data)
            x = make_input(sample)
            ygold = make_output(sample, p2i)
            ypred = predict(w,convert(atype,x))
            ysoft = softmax(ypred)
            ynorm = ysoft ./ priors # acoustic model
            ynorm = Array(ynorm)

            # viterbi
            prob, track = viterbi(ynorm,posteriors)
            solutions[getid(i)] = Dict("logprob"=>prob, "states"=>track)

            if i % o[:period] == 0
                println("$i generation so far ", now())
            end
        end
    end

    ff = open(abspath(o[:savefile]), "w")
    write(ff, JSON.json(solutions))
    close(ff)
    println("done")
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
