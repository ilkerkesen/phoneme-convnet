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
        ("--savefile"; default=nothing)
        ("--atype"; default=(gpu()>=0?"KnetArray{Float32}":"Array{Float32}"))
        ("--window"; default=10; arg_type=Int64)
        ("--period"; default=20; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))
    atype = o[:atype]
    # sr = o[:seed] > 0 ? srand(o[:seed]) : srand()

    # load data
    jsondata = JSON.parsefile(o[:jsonfile])
    p2i, i2p = build_vocab(jsondata)

    # make data
    trnids, tstids = [], []
    # trn = make_data(o[:features], jsondata; extra=trnids)
    tst = make_data(o[:features], jsondata, "test"; extra=tstids)

    # load model
    w = load(o[:model], "w")

    # convert atype
    w = map(wi->convert(atype,wi), w)

    solutions = Dict()

    data = tst
    getid(i) = tstids[i]
    for (i,sample) in enumerate(data)
        x = make_input(sample)
        # ygold = make_output(sample, p2i)
        ygold = map(si->si[2], sample)
        ygold = make_ygold(ygold)
        ypred = predict(w,convert(atype,x))
        ysoft = softmax(ypred)

        generation = generate(ysoft, o[:window])
        generation = map(t->i2p[t], generation)
        solutions[getid(i)] = Dict("ygold"=>ygold, "ypred"=>generation)

        # println("$i generation so far.")
        if i % o[:period] == 0
            println("$i generation so far ", now())
        end
    end

    ff = open(abspath(o[:savefile]), "w")
    write(ff, JSON.json(solutions))
    close(ff)
    println("done")
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
