using SpikeTrainUtilities
using Documenter

DocMeta.setdocmeta!(SpikeTrainUtilities, :DocTestSetup, :(using SpikeTrainUtilities); recursive=true)

makedocs(;
    modules=[SpikeTrainUtilities],
    authors="Dylan Festa",
    repo="https://github.com/dylanfesta/SpikeTrainUtilities.jl/blob/{commit}{path}#{line}",
    sitename="SpikeTrainUtilities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://dylanfesta.github.io/SpikeTrainUtilities.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/dylanfesta/SpikeTrainUtilities.jl",
    devbranch="main",
)
