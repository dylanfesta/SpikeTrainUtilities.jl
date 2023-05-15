using SpikeTrainUtilities
using Documenter

DocMeta.setdocmeta!(SpikeTrainUtilities, :DocTestSetup, :(using SpikeTrainUtilities); recursive=true)

makedocs(;
    modules=[SpikeTrainUtilities],
    authors="Dylan Festa",
    repo="https://github.com/festad/SpikeTrainUtilities.jl/blob/{commit}{path}#{line}",
    sitename="SpikeTrainUtilities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://festad.github.io/SpikeTrainUtilities.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/festad/SpikeTrainUtilities.jl",
    devbranch="main",
)
