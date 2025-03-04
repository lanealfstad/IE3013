# an implementation of value iteration to solve for the optimal policy
# for the recycling robot that is described on page 52 of the RL book.

S = [1,2]    # state is lo, hi
A = [1,2,3]  # action is search, wait, recharge

# transition probabilities when searching
# when action = search
#       lo   hi
#    -------------
# lo |  β   1-β
# hi |1-α    α
α = 0.2
β = 0.3
r_search = 5.0   # expected reward when searching
r_wait = 1.0

# the 4-argument transition probabilities p(sprime,r|s,a).
# let's try to get by with the 3-argument probabilities p(sprime|s,a)
# that are shown in the table on page 52.
# the indexing is p[s, a, sprime]
p3 = zeros(2, 3, 2)
p3[2,1,2] = α
p3[2,1,1] = 1-α
p3[1,1,2] = 1-β
p3[1,1,1] = β
p3[2,2,2] = 1
p3[1,2,1] = 1
p3[1,3,2] = 1
# now note that when the state=hi, action=recharge will not be considered.
# we will take care of that in the value iteration function.

# an implementation of the value iteration algorithm shown on page
# 83 in the RL book.
function value_iteration(V, π)
    γ = 0.9    # discount factor
    θ = 1e-6   # tolerance for convergence
    k = 0      # ieration/sweep
    while true
        δ = 0.0
        for s = 1:2    # loop through the states
            v = V[s]   # old value
            q = zeros(length(A))
            for a in A
                if s==2 && a==3  # s=hi and a=re is not considered
                    continue
                end
                for sprime in S
                    # set the reward. look at the diagram on page 52
                    # and recall that s=lo,hi and that a=se,wa,re.
                    # note that we don't need to loop over the rewards
                    # because there is only one reward associated with
                    # each s,a,sprime.
                    if s==1 && a==1 && sprime==1
                        r = r_search
                    elseif s==1 && a==1 && sprime==2
                        r = -3.0
                    elseif s==1 && a==2 && sprime==1
                        r = r_wait
                    elseif s==1 && a==3 && sprime == 2
                        r = 0.0
                    elseif s==2 && a==1 && sprime==2
                        r = r_search
                    elseif s==2 && a==1 && sprime==1
                        r = r_search
                    elseif s==2 && a==2 && sprime==2
                        r = r_wait
                    else
                        r = 0.0
                    end
                    # make the update
                    q[a] += p3[s,a,sprime]*(r + γ*V[sprime])
                end
            end
            V[s], greedy_action = findmax(q)
            π[s, greedy_action] = 1
            π[s, setdiff(A, greedy_action)] .= 0.0
            δ = max(δ, abs(v - V[s]))
        end
        k += 1
        if δ < θ
            break
        end
    end
    println(k, " sweeps.")
    return V, π
    # output the optimal value function and a deterministic policy
end

# the policy: rows correspond to states, columns correspond to actions.
π = fill(0, length(S), length(A))
V = zeros(length(S))  # will hold the value function
V, π = value_iteration(V, π)

# the optimal policy is
# when the state is lo ==> recharge
# when the state is hi ==> search
