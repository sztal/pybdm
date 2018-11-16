# 2D Block Decomposition Method implementation in Python
# Copyright (c) 2018 Gabriel Goren
# 
# Based on the paper "A Decomposition Method for Global Evaluation of Shannon
# Entropy and Local Estimations of Algorithmic Complexity", by Zenil et al. in
# Entropy 2018, 20(8), 605; https://doi.org/10.3390/e20080605 
# and in previous code by Hector Zenil and Fernando Soler-Toscano
#
# No boundary conditions are implemented (the leftovers after block
# decomposition are ignored in the BDM calculation).
# 
# Usage:
#     python BDM.py [input-file] [lookup-file]
#
# If no argument is provided, the script shows an example run.
# If one argument is provided, it is used as input file.
# If two arguments are provided, the first one is the input file while the
# second one is a custom lookup table that replaces the default one (base
# submatrices of size 4 x 4).
#
#
#
# The MIT License
#
# Permission is hereby granted, free of charge, 
# to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to 
# deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom 
# the Software is furnished to do so, 
# subject to the following conditions:

# The above copyright notice and this permission notice 
# shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR 
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


from collections import Counter
import numpy as np
import sys

def build_lookup_table(filename):
    with open(filename,'r') as datafile:
        lookup = {}
        for line in datafile.readlines():
            base_matrix, kval = line.split(',')
            kval = float(kval)
            lookup[base_matrix] = kval
    return lookup

def string_to_nestedlist(string):
    return [[int(s) for s in row] for row in string.split('-')]

def nestedlist_to_string(nestedlist):
    return '-'.join([''.join([str(i) for i in row]) for row in nestedlist])

def calculate_bdm(string, lookup, d=4, verbose=False):
    """
    Implementation of the 2-dimensional Block Decomposition Method (BDM) for
    the approximation of algorithmic information (also known as algorithmic
    information content or Kolmogorov complexity) of binary matrices.
    
    The matrix is subdivided into d x d base submatrices, each of wich will be
    assigned an algorithmic information content through a lookup table (the table
    itself will have been constructed through the Coding Theorem Method (CTM)).
    The output value for matrix M is 
    
    BDM(M) = sum_i (CTM(m_i) + log_2(n_i))
    
    where i indexes the unique base submatrices m_i that appear in the
    decomposition, and n_i is the number of times each of these submatrices is used.

    After all possible d x d submatrices are fitted inside the original matrix,
    some leftover rows and columns might remain. This issue can be addressed through
    different boundary condition strategies but here the leftovers are simply ignored.
    
    Input
    --------
    string : string
        binary matrix in string format, where rows are separated by minus signs
    lookup : dict
        key is base matrix, value is its CTM value. Base matrices must be of
        size d x d.
    d : integer
        size of base matrices.
    verbose : bool
        If True, prints input matrix, the submatrices into which it is decomposed, and
        the number of times each of these submatrices appear in the original matrix.
        
    Output
    ------
    bdm : float
        BDM value for the matrix.
    """
    
    rows = string_to_nestedlist(string)
    if verbose:
        print('The full matrix to be decomposed:')
        print(np.array(rows))
    nrows, ncols = len(rows), len(rows[0])
    
    for row in rows:
        assert len(row) == ncols, 'The string does not represent a true matrix'
    
    n_rowblocks = int(nrows / d)
    n_colblocks = int(ncols / d)
    
    # Leftovers will be ignored
        
    submatrices = []
    for i in range(n_rowblocks):
        for j in range(n_colblocks):
            # pick a submatrix in list-of-lists format
            submatrix = [row[d * j : d * (j+1)] for row in rows[d * i : d * (i+1)]]                
            # convert to string format
            string = nestedlist_to_string(submatrix)
            submatrices.append(string)
    counts = Counter(submatrices)
    bdm_value = sum(lookup[string] + np.log2(n) for string, n in counts.items())
    
    if verbose:
        print('Base submatrices:')
        for s, n in counts.items():
            submatrix = string_to_nestedlist(s)
            print(np.array(submatrix), 'n =', n)
    
    ## Check whether there were repetitions in the decomposition
    # print('Were all submatrices unique? Answer:', set(counts.values()) == {1})
    
    print('Computed BDM value:', bdm_value)
    return bdm_value
    

def import_stringlist(filename):
    strings = []
    with open(filename,'r') as input_file:
        for line in input_file.readlines():
            string = line[:-1] if line.endswith('\n') else line
            strings.append(string)
    return strings

if __name__ == '__main__':
    narg = len(sys.argv)
    if narg < 3:
        lookup = build_lookup_table('D5.CSV')
        if narg == 1:
            print('Script was called without arguments. Showing example run.', '\n')
            print('BDM values for matrices in input.txt')
            print('------------------------------------')
            strings = import_stringlist('input.txt')
            for s in strings:
                print('Input matrix:')
                print(np.array(string_to_nestedlist(s)))
                value = calculate_bdm(s, lookup)
            
            print('\n', 'Some other examples, with verbose output:')
            print('-----------------------------------------')
            string = '-'.join(['10011001' * 4 for i in range(10)])
            calculate_bdm(string, lookup, verbose=True)
            string = '-'.join(['1001001' * 4 for i in range(10)])
            calculate_bdm(string, lookup, verbose=True)

        else:
            print('Script was called with an input file.')
            print('Using default lookup-table (4 x 4 base matrices)')
            filename = sys.argv[1]
            strings = import_stringlist(filename)
            for s in strings:
                value = calculate_bdm(s, lookup)
            
    else:
        print('Script was called with input file and custom lookup-table.')
        filename = sys.argv[1]
        lookup = build_lookup_table(sys.argv[2])
        strings = import_stringlist(filename)
        for s in strings:
            value = calculate_bdm(s, lookup)
    
    
