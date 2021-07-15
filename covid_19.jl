using Random,
    NearestNeighbors,
    Distributions,
    Base.Threads,
    DelimitedFiles,
    Printf,
    Base.Filesystem
    # ProgressMeter

function near_calc(location, A, B, ϵ)
    tree = KDTree(location[:, B])
    inrange(tree, location[:, A], ϵ, false)
end

function main(seed_number, popower = 5)
    Random.seed!(seed_number)
   
    println("\n                          seed: $seed_number")
    iᵣ = rand(Uniform(0.01,0.02))
    iᵣₛ = rand(Uniform(0.01,0.02))
    println("ir: $(round(iᵣ, digits = 4)), irs: $(round(iᵣₛ, digits = 4))")
    
    pop = 10^popower # 기본 6승
    location = domain_size * rand(2, pop)
    first_infected_time = zeros(Int, pop)

    # E_ = zeros(Int64, end_time)
    # I_ = zeros(Int64, end_time)
    # R_ = zeros(Int64, end_time)
    
    state = zeros(Int, pop)
    state[1] = 1

    data = zeros(Int64, end_time, 4)
    E_time = rand(Weibull(2.37348, 6.544), pop)
    I_time = E_time + rand(Weibull(2.30055, 9.48721), pop)
    speed = abs.(randn(pop))

    for i = 1:end_time
        S = state .== 0
        E = state .== 1
        I = state .== 2
        R = state .== 3
        # print()
        # println([j i count(S) count(E) count(I) count(R)])
        data[i, :] = [count(S) count(E) count(I) count(R)]
        print("$(count(I)) - ")
        if count(E) + count(I) == 0
            for k = 1:4
                data[i:end,k] .= data[i,k]
            end
            break
        end
        # 여기서 자료 저장
        location += [speed .* randn(size(location, 2)) speed .* randn(size(location, 2))]'
        location = mod.(location, domain_size)
        # 여기서 위치 이동
        near1 = length.(near_calc(location, S, E, ϵ))
        near2 = length.(near_calc(location, S, I, ϵ))
        death = rand(count(S)) .> ((1 - (iᵣₛ)) .^ near1) .* ((1 - iᵣ) .^ near2)
        state[S] += death
        first_infected_time[S] = death .* i
        transition_time_E_to_I = first_infected_time + E_time
        transition_time_I_to_R = first_infected_time + I_time
        # 여기서 상태 변화

        state[E.&(transition_time_E_to_I.<i)] .= 2
        state[I.&(transition_time_I_to_R.<i)] .= 3
    end
    # writedlm(
    #     "C:\\Users\\rlarb\\Desktop\\code\\data_with_E_" * string(iᵣₛ) * "\\" * string(j) * ".csv",
    #     data,
    #     ',',
    # )
    if data[10,3] != 0
        notepad_I = open("training_I $popower.csv", "a")
        notepad_R = open("training_R $popower.csv", "a")
        try
            println(notepad_I, "$seed_number, $iᵣ, $iᵣₛ, ", string(data[:,3])[2:end-1])
            println(notepad_R, "$seed_number, $iᵣ, $iᵣₛ, ", string(data[:,4])[2:end-1])
        catch
            println("error: something wrong!")
        finally
            close(notepad_I)
            close(notepad_R)
        end
        print(">")
    else
        print("!")
    end
end


#-------


const domain_size = 100 # 기본 100
const end_time = 10^2
const ϵ = 0.5
N = 10000
# p = Progress(N);
# update!(p, 0)
# jj = Atomic{Int}(0)
# l = SpinLock()

cd(@__DIR__); println(pwd())

# for j in [0.004, 0.006, 0.007, 0.008, 0.009]
for popower in [7]
    notepad_I = open("training_I $popower.csv", "w")
    notepad_R = open("training_R $popower.csv", "w")
    print(notepad_I, "obs, ir, irs")
    print(notepad_R, "obs, ir, irs")
    for t ∈ 1:end_time
        print(notepad_I, ", t$t")
        print(notepad_R, ", t$t")
    end
    println(notepad_I)
    println(notepad_R)
    close(notepad_I)
    close(notepad_R)
    # if !isdir("C://Users//rlarb//Desktop//code//data_with_E_" * string(j))
    #     mkdir("C://Users//rlarb//Desktop//code//data_with_E_" * string(j))
    # end
    # file_list = readdir("C://Users//rlarb//Desktop//code//data_with_E_" * string(j))
    # for i = 1:N
    for seed_number = 1:N
        # if string(i) * ".csv" ∉ file_list
            main(seed_number, popower)
        # end
        # atomic_add!(jj, 1)
        # lock(l)
        # update!(p, jj[])
        # unlock(l)
    end
end