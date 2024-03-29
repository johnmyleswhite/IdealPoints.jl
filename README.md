IdealPoints.jl
==============

This package is unofficial and intended for my own personal use. That said, it
is reasonably well-tested and reasonably well-documented, so I'm making it
publicly available with the understanding that I do not intend to maintain this
code for anyone else's use.

Usage example:

```jl
import RollCallDataIO: SparseRollCall
import IdealPoints: ideal_points
import DataFrames: DataFrame, writetable

path = joinpath("data", "senate", "112.ord")
io = open(path, "r")
ord_file = read(RollCallDataIO.ORDFile, io)
close(io)

ds = 2
σ_x = 0.25
σ_y = 10.0

a, b, c, d = ideal_points(ord_file, ds, σ_x, σ_y, 100, true)

n_legislators = length(RollCallDataIO.legislators(ord_file))

df = DataFrame(
    x1 = transpose(reshape(a, ds, n_legislators))[:, 1],
    x2 = transpose(reshape(a, ds, n_legislators))[:, 2],
    legislator = RollCallDataIO.legislators(ord_file),
    party = RollCallDataIO.parties(ord_file),
)

writetable("ideal_points.csv", df)
```
