# Programmer Jovan Araiza 11469149
import operator as op
from functools import reduce


# Constants
PAIR = 2
DEBUG = False

# Functions
def ncr(n, r):
    '''
    This function was retrieved from stackoverflow, computes n choose r
    '''
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

def load_data(location = "./BrowsingData.txt"):
    f = open(location,"r")
    baskets = [line.split(" ")[:-1] for line in f]
    f.close()
    return baskets

def a_priori(baskets,S,outfile):
    '''Implementation of A_priori algorithm

    Arguments:
        baskets {list of list} -- Basket data
        S {int} -- Support Threshold
        outfile {file} -- Output is written to this file
    '''

    # Triangular-Matrix function
    k = lambda i,j,n: int((i-1)*(n- i/2)+j-i)

    # Count items
    Item_mapping = {}
    Item_Count = {}

    for basket in baskets:
        for item in basket:
            Item_Count[item] = Item_Count.get(item,0)+1

    # Pass 1 : Determine frequent items
    frequent_items = set()
    for item, count in Item_Count.items():
        if count >= S:
            frequent_items.add(item)

    if DEBUG:
        print("\nFrequent Items: ")
        for item in frequent_items:
            print(item)

    # Total unique frequent items
    n = len(frequent_items)

    # Creating ragged array
    ragged_array = [0]*(ncr(n,PAIR)+1)

    # Map items to integers
    for item, index in list(zip(frequent_items,range(n))):
        Item_mapping[item] = index+1

    # Pass 2 : Count pairs in ragged array
    pairs = set()
    for basket in baskets:
        f_items = set([item for item in basket if item in frequent_items])
        for item_i in f_items:
            for item_j in f_items:
                i = Item_mapping[item_i]
                j = Item_mapping[item_j]
                if i < j:
                    pairs.add((item_i,item_j))
                    ragged_array[k(i,j,n)]+=1

    # don't need this anymore
    del frequent_items

    # Determine frequent pairs
    frequent_pairs = set()
    for item_i,item_j in pairs:
        i = Item_mapping[item_i]
        j = Item_mapping[item_j]
        if ragged_array[k(i,j,n)] >= S:
            frequent_pairs.add((item_i,item_j))

    # Don't need this anymore
    del pairs

    if DEBUG:
        print("\nFrequent Pairs:")
        for pair in frequent_pairs:
            print(pair)

    # Determine confidence of association for pairs
    frequent_pair_confidence = []
    for item_i,item_j in frequent_pairs:
        i = Item_mapping[item_i]
        j = Item_mapping[item_j]

        conf = ragged_array[k(i,j,n)] / Item_Count[item_i]
        frequent_pair_confidence.append((item_i,item_j,conf))

        conf = ragged_array[k(i,j,n)] / Item_Count[item_j]
        frequent_pair_confidence.append((item_j,item_i,conf))

    # Don't need this anymore
    del Item_Count

    Key = lambda x: x[-1]

    # Sort and print to file
    frequent_pair_confidence = sorted(frequent_pair_confidence,key=Key,reverse=True)
    print("\nOUTPUT A",file=outfile)
    for i in range(5):
        i,j,conf = frequent_pair_confidence[i]
        print("%s => %s %f"%(i,j,conf),file=outfile)

    # Don't need this anymore
    del frequent_pair_confidence
    # Count frequent triples
    triples = set()
    Item_Count_triples = {}

    frequent_pairs_flat = set()
    for i,j in frequent_pairs:
        frequent_pairs_flat.add(i)
        frequent_pairs_flat.add(j)

    # Don't need this anymore
    del frequent_pairs

    for basket in baskets:
        # Only count those of frequent pairs
        f_items = set([item for item in basket if item in frequent_pairs_flat])
        for item_i in f_items:
            for item_j in f_items:
                for item_z in f_items:
                    i = Item_mapping[item_i]
                    j = Item_mapping[item_j]
                    z = Item_mapping[item_z]
                    if i < j and j < z:
                        new_triple = (item_i,item_j,item_z)
                        triples.add(new_triple)
                        Item_Count_triples[new_triple] = Item_Count_triples.get(new_triple,0)+1

    # Determine frequent triples
    frequent_triples = set()
    for triple in triples:
        if Item_Count_triples[triple] >= S:
            frequent_triples.add(triple)
    if DEBUG:
        print("\nFrequent Triples:")
        for triple in frequent_triples:
            print(triple)

    # Determine confidence of association for triples
    frequent_triple_confidence = []
    for item_i,item_j,item_z in frequent_triples:
        i = Item_mapping[item_i]
        j = Item_mapping[item_j]
        z = Item_mapping[item_z]

        conf = Item_Count_triples[(item_i,item_j,item_z)] / ragged_array[k(i,j,n)]
        frequent_triple_confidence.append((item_i,item_j,item_z,conf))

        conf = Item_Count_triples[(item_i,item_j,item_z)] / ragged_array[k(i,z,n)]
        frequent_triple_confidence.append((item_i,item_z,item_j,conf))

        conf = Item_Count_triples[(item_i,item_j,item_z)] / ragged_array[k(j,z,n)]
        frequent_triple_confidence.append((item_j,item_z,item_i,conf))

    frequent_triple_confidence = sorted(frequent_triple_confidence,key=Key,reverse=True)

    r = set()
    if len(frequent_triple_confidence) >= 5:
        for triple in frequent_triple_confidence[:5]:
            r.add(triple[-1])
        if len(r) == 1:
            frequent_triple_confidence = sorted(frequent_triple_confidence[:5],key=lambda x: x[0])

    print("\nOUTPUT B",file=outfile)
    for i in range(5) if len(frequent_triple_confidence)> 5 else range(len(frequent_triple_confidence)):
        i,j,z,conf = frequent_triple_confidence[i]
        print("%s %s => %s %f"%(i,j,z,conf),file=outfile)

if __name__ == "__main__":
    #location = "./BrowsingData50.txt"
    location = "./BrowsingData.txt"

    baskets = load_data(location)

    #S = 8
    S = 100

    outfile = open("output.txt","w")
    print("Using dataset '%s' with support threshold %d"%(location,S),file=outfile)

    a_priori(baskets,S,outfile)
    outfile.close()