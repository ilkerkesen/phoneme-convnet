using ArgParse
using JLD
using MAT
using JSON
using HDF5
using MFCC

include("util.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "MFSC feature extraction from MAT file"

    @add_arg_table s begin
        ("--matfile"; required=true)
        ("--jsonfile"; required=true)
        ("--savefile"; required=true)
        ("--sr"; default=16000.0; arg_type=Float64)
        ("--preprocess"; action=:store_true)
        ("--period"; default=100; arg_type=Int64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # load mat file
    data = matread(o[:matfile])
    entries = JSON.parsefile(o[:jsonfile])
    savefile = h5open(o[:savefile], "w")

    println("started.", now())
    @time for (i,entry) in enumerate(entries)
        longid = entry["longid"]
        sample = data[longid]
        feats = mfsc(sample)
        write(savefile, longid, feats[1])

        if i % o[:period] == 0
            println("$i samples processed so far...")
        end
    end
    close(savefile)
    println("done.")
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
