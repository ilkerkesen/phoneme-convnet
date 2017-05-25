using ArgParse
using JLD,MAT,JSON

include("util.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "CNN-HMM hybrid ASR system in Knet by Ilker Kesen."

    @add_arg_table s begin
        ("--jsonfile"; required=true)
        ("--savefile"; required=true)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    jsondata = JSON.parsefile(abspath(o[:jsonfile]))
    p2i, i2p = build_vocab(jsondata)

    posteriors, priors = get_probabilities(jsondata, p2i, i2p)
    save(o[:savefile], "posteriors", posteriors, "priors", priors)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
