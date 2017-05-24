using ArgParse
using JLD
using MAT
using JSON
using HDF5

function main(args)
    s = ArgParseSettings()
    s.description = "MFSC feature extraction from MAT file"

    @add_arg_table s begin
        ("--matfile"; required=true)
        ("--jsonfile"; required=true)
        ("--savefile"; required=true)
        ("--sr"; default=16000.0; arg_type=Float64)
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    o[:atype] = eval(parse(o[:atype]))

    # load mat file
    data = matread(o[:matfile])
    entries = JSON.parsefile(o[:jsonfile])
    file = h5open(o[:savefile], "w")

    println("started.", now())
    @time for entry in entries
        longid = entry["longid"]
        sample = data[longid]
        feats = mfsc(sample, o[:sr])
        write(file, longid, feats)
    end
    close(file)
    println("done.")
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
