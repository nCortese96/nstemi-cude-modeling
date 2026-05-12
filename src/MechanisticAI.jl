"""
MechanisticAI.jl

Central include entrypoint for the refactored script copies. During this first
cleanup pass it deliberately avoids wrapping code in a module so existing helper
names and the integrated `MultiStartOptimizer` module remain available exactly
as before.
"""

include("helpers.jl")
