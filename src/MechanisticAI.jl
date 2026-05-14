"""
MechanisticAI.jl

Central include entrypoint for the refactored script copies. During this
transition pass it deliberately avoids wrapping code in a module so existing
helper names remain available while helpers are split into focused files.
"""

include("data_io.jl")
include("preprocessing.jl")
include("models.jl")
include("fitting.jl")
include("helpers.jl")
