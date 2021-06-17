@time using Statistics
@time using Random
@time using Base.Threads
@time using Dates
@time using NearestNeighbors
# @time using LightGraphs
@time using LinearAlgebra
@time using Distributions
# @time using Plots
# @time using CUDA
@time using CSV, DataFrames

r2(x) = round(x, digits = 2)

# seed_number = 0
# Random.seed!(seed_number);

n = 10^5
ID = 1:n
number_of_host = 1
# para_range = Uniform(0.05, 0.20)
ε = 5*10^(-3)
brownian = MvNormal(2, ε) # moving process
end_time = 100

notepad_I = open("training_I.csv", "w")
notepad_R = open("training_R.csv", "w")
print(notepad_I, "obs, bom, beta, mu")
print(notepad_R, "obs, bom, beta, mu")
for t ∈ 1:end_time
    print(notepad_I, ", t$t")
    print(notepad_R, ", t$t")
end
println(notepad_I)
println(notepad_R)
close(notepad_I)
close(notepad_R)

for T ∈ 1:10^7
println("                             T: $T")
S_ = zeros(Int64, end_time)
I_ = zeros(Int64, end_time)
R_ = zeros(Int64, end_time)
state = Array{Char, 1}(undef, n); state .= 'S' # using SIR model
host = rand(ID, number_of_host); state[host] .= 'I'

location = rand(2, n) # micro location
# β, μ = 0.11492613, 0.16816714
β, μ = rand(Uniform(0.1, 0.9)), rand(Uniform(0.1, 0.9))
R_0 = β / μ
println("β: $(r2(β)), μ: $(r2(μ)), R_0: $(r2(R_0))")
if R_0 > 2 println("bom > 2!"); continue; end

for t ∈ 1:end_time

    bit_S = (state .== 'S'); n_S = sum(bit_S); S_[t] = n_S
    bit_I = (state .== 'I'); n_I = sum(bit_I); I_[t] = n_I
    bit_R = (state .== 'R'); n_R = sum(bit_R); R_[t] = n_R

    print("$n_I - ")
    # println("t: $(lpad(t, 3, '0')) ",
    #         "S: $(lpad(n_S, 6, '_')) ",
    #         "I: $(lpad(n_I, 6, '_')) ",
    #         "R: $(lpad(n_R, 6, '_'))")

    if n_I == 0
        println("!")
        break
    end
    if t == end_time
        println(">")
        break
    end

    ID_S = ID[bit_S]
    ID_I = ID[bit_I]

    location[:,ID_S] = mod.(location[:,ID_S] + rand(brownian, n_S), 1.0)
    location[:,ID_I] = mod.(location[:,ID_I] + rand(brownian, n_I), 1.0)

    kdtreeI = KDTree(location[:,ID_I])
    contact = length.(inrange(kdtreeI, location[:,ID_S], ε))

    bit_infected = rand(n_S) .< (1 .- (1 - β).^contact)
    ID_infected = ID_S[bit_infected]

    state[ID_infected] .= 'I'
    state[(bit_I .& (rand(n) .< μ))] .= 'R'
end

if I_[10] != 0
    notepad_I = open("training_I.csv", "a")
    notepad_R = open("training_R.csv", "a")
    try
        println(notepad_I, "$T, $(β/μ), $β, $μ, ", string(I_)[2:end-1])
        println(notepad_R, "$T, $(β/μ), $β, $μ, ", string(R_)[2:end-1])
    catch
        println("error: something wrong!")
    finally
        close(notepad_I)
        close(notepad_R)
    end
end

end
