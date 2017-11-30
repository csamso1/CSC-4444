def main():
    num_symbols = input("Enter the number of distinct propositional symbols: ")
    print("number of distinct propositional symbols = " + num_symbols)
    num_clauses = input("Enter the number of clauses in S: ")
    print("number of clauses in S: " + num_clauses)
    num_litterals = input("Enter the maximum number of literals in a clause in S: ")
    print("The maximum number of literals in a clause in S = " + num_litterals)
    negative_litteral_prob = float(input("Enter the probability (between .4 and .6) that a litteral will be negative: "))
    while(negative_litteral_prob < float(.4) or negative_litteral_prob > float(.6)):
        negative_litteral_prob = float(input("The probability must be between .4 and .6, please provide a valid probability: "))
    print('The probability that a litteral will be negative is: {} ' .format(negative_litteral_prob))

def litteral():
    label = ""
    sign = 






def WalkSAT(clauses, p=0.5, max_flips=10000):
    """Checks for satisfiability of all clauses by randomly flipping values of variables
    """
    # Set of all symbols in all clauses
    symbols = {sym for clause in clauses for sym in prop_symbols(clause)}
    # model is a random assignment of true/false to the symbols in clauses
    model = {s: random.choice([True, False]) for s in symbols}
    for i in range(max_flips):
        satisfied, unsatisfied = [], []
        for clause in clauses:
            (satisfied if pl_true(clause, model) else unsatisfied).append(clause)
        if not unsatisfied:  # if model satisfies all the clauses
            return model
        clause = random.choice(unsatisfied)
        if probability(p):
            sym = random.choice(list(prop_symbols(clause)))
        else:
            # Flip the symbol in clause that maximizes number of sat. clauses
            def sat_count(sym):
                # Return the the number of clauses satisfied after flipping the symbol.
                model[sym] = not model[sym]
                count = len([clause for clause in clauses if pl_true(clause, model)])
                model[sym] = not model[sym]
                return count
            sym = argmax(prop_symbols(clause), key=sat_count)
        model[sym] = not model[sym]
    # If no solution is found within the flip limit, we return failure
    return None

if __name__ == "__main__":main()