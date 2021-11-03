
# parameter p in geometric probability distribution of PMD
const PMDpparam = 0.3

# constant C in geometric probability distribution of PMD
const PMDconstant = 0.01

# True biological polymorphism between the ancient individual and the reference sequence
const polymorphism_ancient = 0.001

# True biological polymorphism between the contaminants and the reference sequence
const polymorphism_contamination = 0.001


function phred2prob(Q)
    q = Q % Int
    return 10.0^(-q / 10.0)
end

function phreds2probs(Qs)
    return phred2prob.(Qs)
end

function damage_model_modern(z)
    return 0.001
end

function damage_model_ancient(z, p = PMDpparam, C = PMDconstant)
    return Dz = p * (1 - p)^(z - 1) + C
end

function L_match(z, damage_model, quality, polymorphism)
    P_damage = damage_model(z)
    P_error = phred2prob(quality) / 3
    P_poly = polymorphism
    P_match =
        (1.0 - P_damage) * (1.0 - P_error) * (1.0 - P_poly) +
        (P_damage * P_error * (1.0 - P_poly)) +
        (P_error * P_poly * (1.0 - P_damage))
    return P_match
end

function L_mismatch(z, damage_model, quality, polymorphism)
    P_match = L_match(z, damage_model, quality, polymorphism)
    P_mismatch = 1 - P_match
    return P_mismatch
end


function compute_PMD_score(
    sequence::LongDNASeq,
    reference::LongDNASeq,
    qualities::Vector{UInt8},
    max_position::Int = -1,
)::Float64

    if max_position < 1
        max_position = length(sequence)
    end

    L_D = 1.0
    L_M = 1.0

    z = 1
    for (s, r, q) in zip(sequence, reference, qualities)

        if s == DNA_N | r == DNA_N
            continue
        end

        if r == DNA_C
            if s == DNA_T
                L_D *= L_mismatch(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_mismatch(z, damage_model_modern, q, polymorphism_contamination)
            elseif s == DNA_C
                L_D *= L_match(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_match(z, damage_model_modern, q, polymorphism_contamination)
            end

        elseif r == DNA_G

            if s == DNA_A
                L_D *= L_mismatch(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_mismatch(z, damage_model_modern, q, polymorphism_contamination)
            elseif s == DNA_G
                L_D *= L_match(z, damage_model_ancient, q, polymorphism_ancient)
                L_M *= L_match(z, damage_model_modern, q, polymorphism_contamination)
            end
        end

        z += 1

        if z > max_position
            break
        end

    end

    PMD = log(L_D / L_M)
    return PMD

end
