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
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))


end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
