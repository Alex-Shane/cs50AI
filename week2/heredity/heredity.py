import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    #if len(sys.argv) != 2:
        #sys.exit("Usage: python heredity.py data.csv")
    #people = load_data(sys.argv[1])
    people = load_data("data/family0.csv")
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    probs = []
    no_gene = set(list(people.keys()))-one_gene-two_genes
    for person in people:
        if people[person]['mother'] == None:
            if person in no_gene:
                copy_prob = PROBS['gene'][0]
                trait_prob = PROBS['trait'][0][person in have_trait]
            elif person in one_gene:
                copy_prob = PROBS['gene'][1]
                trait_prob = PROBS['trait'][1][person in have_trait]
            else:
                copy_prob = PROBS['gene'][2]
                trait_prob = PROBS['trait'][2][person in have_trait]
            probs.append(copy_prob*trait_prob)
        else:
            mom = people[person]['mother']
            dad = people[person]['father']
            if person in no_gene:
                no_mom = 0
                if mom in no_gene:
                    no_mom = 1-PROBS['mutation']
                elif mom in one_gene:
                    no_mom = 0.5 
                else:
                    no_mom = PROBS['mutation']
                no_dad = 0
                if dad in no_gene:
                    no_dad = 1-PROBS['mutation']
                elif dad in one_gene:
                    no_dad = 0.5 
                else:
                    no_dad = PROBS['mutation']
                probs.append(no_mom*no_dad)
            elif person in one_gene:
                from_mom = 0
                if mom in no_gene:
                    from_mom = PROBS['mutation']
                elif mom in one_gene:
                    from_mom = 0.5 
                else:
                    from_mom = 1-PROBS['mutation']
                from_dad = 0
                if dad in no_gene:
                    from_dad = PROBS['mutation']
                elif dad in one_gene:
                    from_dad = 0.5
                else:
                    from_dad = 1-PROBS['mutation']
                not_from_mom = 1-from_mom
                not_from_dad = 1-from_dad
                probs.append(from_mom*not_from_dad + from_dad*not_from_mom)
            else:
                from_mom = 0
                if mom in no_gene:
                    from_mom = PROBS['mutation']
                elif mom in one_gene:
                    from_mom = 0.5 
                else:
                    from_mom = 1-PROBS['mutation']
                from_dad = 0
                if dad in no_gene:
                    from_dad = PROBS['mutation']
                elif dad in one_gene:
                    from_dad = 0.5 
                else:
                    from_dad = 1-PROBS['mutation']
                probs.append(from_mom*from_dad)
    probability = 1
    for num in probs:
        probability *= num
    return probability
        

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        num = 1 if person in one_gene else 2 if person in two_genes else 0
        probabilities[person]['gene'][num] += p
        probabilities[person]['trait'][person in have_trait] += p
    
        


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        total_gene = sum(list(probabilities[person]['gene'].values()))
        for key in probabilities[person]['gene']:
            probabilities[person]['gene'][key] /= total_gene
        total_trait = sum(list(probabilities[person]['trait'].values()))
        for key in probabilities[person]['trait']:
            probabilities[person]['trait'][key] /= total_trait


if __name__ == "__main__":
    main()
