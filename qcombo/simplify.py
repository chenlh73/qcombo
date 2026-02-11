from sympy.combinatorics.permutations import Permutation
from sympy.tensor.indexed import Indexed
from sympy import IndexedBase,simplify
from sympy.core.mul import Mul
from sympy.core.add import Add
from .tools import ProgressBar

#######################################################################
# Simplify using antisymmetry properties
def sort_indices(indices):
    '''
    Sort a set of indices and return the sorted indices along with the sign
    
    Parameters:
        indices: index tuple
        
    Returns:
        (sorted_indices, sign): sorted indices and sign determined by number of swaps
    ## Example:
    input: (a,c)  >>  output : (a,c), 1
    input: (b,a)  >>  output : (a,b), -1
    '''
    # Convert indices to list
    index_list = list(indices)
    
    # Sort alphabetically
    sorted_list = sorted(index_list, key=str)
    
    # Calculate permutation sign
    if index_list == sorted_list:
        sign = 1
    else:
        # Create mapping from original positions to sorted positions
        permutation = []
        for elem in sorted_list:
            permutation.append(index_list.index(elem))
        
        # Calculate parity of permutation
        p = Permutation(permutation)
        sign = 1 if p.is_even else -1
    
    sorted_tuple = tuple(sorted_list)

    return sorted_tuple, sign

def sort_indexed_tensor(tensor):
    """
    Sort upper and lower indices of an Indexed tensor separately, and return sorted tensor and coefficient
    
    Parameters:
        tensor: sympy.tensor.indexed.Indexed object
        
    Returns:
        (sorted_tensor, coefficient): sorted tensor and coefficient determined by number of swaps
    ## Example:
    input: G[(c,a),(e,f)]  >>  output : G[(a,c),(e,f)], -1
    input: G[(c,a),(f,e)]  >>  output : G[(a,c),(e,f)], 1
    """
    if not isinstance(tensor, Indexed):
        raise ValueError("Input must be an Indexed object")
    
    # Get all index groups of tensor
    index_groups = tensor.indices

    total_sign = 1
    
    # Separate upper and lower index groups
    upper_indices = index_groups[0]
    lower_indices = index_groups[1]

    new_upper_indices, upper_sign = sort_indices(upper_indices)
    new_lower_indices, lower_sign = sort_indices(lower_indices)

    sorted_tensor = tensor.base[new_upper_indices, new_lower_indices]
    total_sign *= upper_sign * lower_sign

    return sorted_tensor, total_sign

def sort_mul_expression(expr):
    """
    Reorder indices for each Indexed object in a Mul expression
    
    Parameters:
        expr: sympy.core.mul.Mul object
        
    Returns:
        sorted_expr: sorted expression
    ## Example:
    input:  G[(c,a),(e,f)]*H[(f,e),(b,a)]   
    >> output: -G[(a,c),(e,f)]*H[(e,f),(a,b)]
    """
    if not isinstance(expr, Mul):
        raise ValueError("Input must be a Mul object")
    
    # Initialize total coefficient
    total_coefficient = 1
    
    # Store processed factors
    sorted_factors = []
    
    # Process each factor
    for tensor in expr.args:
        if isinstance(tensor, Indexed):
            # Get tensor name
            sorted_tensor, coefficient = sort_indexed_tensor(tensor)
            sorted_factors.append(sorted_tensor)
            total_coefficient *= coefficient
        else:
            # Non-Indexed objects added directly
            sorted_factors.append(tensor)
    
    # Build new expression
    sorted_expr = Mul(*sorted_factors) * total_coefficient
    
    return sorted_expr

def sort_add_expression(expr):
    """
    Reorder indices for each term in an Add expression
    
    Parameters:
        expr: sympy.core.add.Add object
        
    Returns:
        sorted_expr: sorted expression
    ## Example:
    input: G[(c,a),(e,f)]*H[(f,e),(b,a)] - G[(a,c),(e,f)]*H[(f,e),(b,a)]  
    >> output: 2*G[(a,c),(e,f)]*H[(e,f),(a,b)]
    """
    if not isinstance(expr, Add):
        raise ValueError("Input must be an Add object")
    
    # Store processed terms
    sorted_terms = []
    
    # Process each term
    progress = ProgressBar(len(expr.args), "Antisymmetry Simplify")
    for term in expr.args:
        if isinstance(term, Mul):
            # Process Mul term
            sorted_term = sort_mul_expression(term)
            sorted_terms.append(sorted_term)
        elif isinstance(term, Indexed):
            # Process single Indexed term
            sorted_term, sign = sort_indexed_tensor(term)
            sorted_terms.append(sorted_term * sign)
        else:
            # Other types of terms added directly
            sorted_terms.append(term)
        progress.update()
    
    # Build new expression
    sorted_expr = Add(*sorted_terms)
    
    return sorted_expr

#######################################################################
# Simplify expressions containing lambda terms using dummy index properties

def reorder_dummy_indices_mul(expr):
    """
    Reorder dummy indices so that lambda indices become the first few indices
    in alphabetical order, excluding A indices
    
    Parameters:
    expr: Mul expression
    
    Returns:
    Re-indexed expression
    """
    if not isinstance(expr, Mul):
        raise ValueError("Input must be a Mul object")
    
    # Extract tensors from expression
    factors = expr.args
    
    A_tensor = None
    lambda_tensor = []  # Can handle multiple lambda variables

    for tensor in factors:
        if isinstance(tensor, Indexed):  # Ensure it's a tensor object
            if tensor.base == IndexedBase('A'):
                A_tensor = tensor
            elif tensor.base == IndexedBase(chr(955)):  # Lambda symbol
                lambda_tensor.append(tensor)

    # If no lambda terms, return original expression
    if len(lambda_tensor) == 0:
        return expr

    # Get A indices (these are fixed indices, not reindexed)
    A_indices = set()
    if A_tensor is not None:  # Zero-body terms won't have A tensor
        for index_tuple in A_tensor.indices:    
            A_indices.update(index_tuple)
    
    # Collect all dummy indices
    all_dummy_indices = set()
    for tensor in factors:
        if isinstance(tensor, Indexed):  # Ensure it's a tensor object
            if tensor.base != IndexedBase('A'):
                for index_tuple in tensor.indices:
                    all_dummy_indices.update(index_tuple)
    
    # Remove A indices (in principle, A should not be here)
    dummy_indices = all_dummy_indices - A_indices
    
    # Sort dummy indices alphabetically
    sorted_dummy_indices = sorted(dummy_indices, key=str)
    
    # Get initial lambda indices
    lambda_flat_indices = []
    for tensor in lambda_tensor:
        for index_tuple in tensor.indices:
            lambda_flat_indices.extend(index_tuple)

    # Determine how many new indices lambda needs
    lambda_index_count = len(lambda_flat_indices) 

    # Assign first few sorted indices to lambda
    new_lambda_indices = sorted_dummy_indices[:lambda_index_count]
    
    # Assign remaining sorted indices to other dummy indices
    remaining_indices = sorted_dummy_indices[lambda_index_count:]

    # Create index mapping
    index_mapping = {}
    
    # First map lambda indices
    for old_idx, new_idx in zip(lambda_flat_indices, new_lambda_indices):
        index_mapping[old_idx] = new_idx
    
    # Then map remaining dummy indices
    # remaining_old_indices = list(dummy_indices - set(lambda_flat_indices))
    remaining_old_indices = sorted(dummy_indices - set(lambda_flat_indices), key=str)  # Reorder remaining dummy indices, update: 2025/11/18
    for old_idx, new_idx in zip(remaining_old_indices, remaining_indices):
        index_mapping[old_idx] = new_idx
    
    # Apply index mapping to all tensors
    new_expr = expr.xreplace(index_mapping)
    
    return new_expr


def reorder_dummy_indices_add(expr):
    """
    Reorder dummy indices so that lambda indices become the first few indices
    in alphabetical order, excluding A indices
    
    Parameters:
    expr: Add expression
    
    Returns:
    Re-indexed expression
    """
    if not isinstance(expr, Add):
        raise ValueError("Input must be an Add object")
    
    # Store processed terms
    reorder_terms = []
    
    # Process each term
    progress = ProgressBar(len(expr.args), "Lambda Simplify")
    
    for term in expr.args:
        reorder_terms.append(reorder_dummy_indices_mul(term))
        progress.update()
            
    # Build new expression
    reorder_expr = Add(*reorder_terms)

    return reorder_expr

#######################################################
# Filter expressions based on different lambda polynomial terms
# add by chenlh, 2025/11/12

def get_Indexed_IndicesNum(tensor):
    """
    Get the number of indices (rank) of a tensor
    
    Parameters:
        tensor: tensor object, can be Indexed object or other objects containing indices
        
    Returns:
        Number of tensor indices (integer)
    """
    # Check if it's a SymPy Indexed object
    if not isinstance(tensor, Indexed):
        raise ValueError("Input must be an Indexed object")
    
    return len(tensor.indices[0]) + len(tensor.indices[1])

def filterLambdaBody(expr, filterLambdaBody):
    """
    Filter terms in expression based on the number of indices in Lambda tensor
    
    Parameters:
        expr: SymPy expression (Add class)
        filterLambdaBody: target Lambda tensor body term, integer
        
    Returns:
        Filtered expression
    
    # Example
    expr = A[(a,b),(c,d)]*lambda[(e),(f)] + A[(a,b),(c,d)]*lambda[(e,f),(g,h)]
    filterLambdaBody = 2
    >> A[(a,b),(c,d)]*lambda[(e,f),(g,h)]
    """
    # Ensure expression is of type Add
    if not isinstance(expr, Add):
        raise ValueError("Expression must be of type Add")
    
    # Filter terms meeting the criteria
    filtered_terms = []
    no_lambda_terms = []

    for mul in expr.args:  # Iterate through each mul class term in Add class
        lambda_found = False
        lambda_body = 0
        for tensor in mul.args:   # Iterate through each indexed class term in mul class
            if isinstance(tensor, Indexed):
                if tensor.base == IndexedBase(chr(955)):  # Lambda symbol
                    lambda_found = True
                    lambda_body = int((get_Indexed_IndicesNum(tensor)) / 2)
                    if lambda_body == filterLambdaBody:
                        filtered_terms.append(mul)
                    break
        # If no lambda found, add to no_lambda_terms
        if not lambda_found:
            no_lambda_terms.append(mul)

    # If filtered lambda body term is 0 or 1, return terms without lambda
    if filterLambdaBody == 0 or filterLambdaBody == 1:
        return Add(*no_lambda_terms)
    # Otherwise return normally
    return Add(*filtered_terms)

# Merge terms with same G and H
# add by chenlh 2025/11/17
# update: 2025/11/20 Fixed bug where it would keep running if no terms could be simplified

def uniteSameGAndH(expr):
    # --
    if expr == 0:
        return expr

    if not isinstance(expr, Add):
        raise ValueError("Input must be an Add object")
    
    united_expr = []

    add_expr = expr

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # When add_expr has only one term, add_expr becomes Mul class, causing errors

    while add_expr != 0:
        # Take G and H from first term for simplification
        # When add_expr has only one term, add_expr becomes Mul class, causing errors
        if isinstance(add_expr, Add):
            first_mul_term = add_expr.args[0]
        elif isinstance(add_expr, Mul):
            first_mul_term = add_expr
        # Find G and H to be merged
        for tensor in first_mul_term.args:
            if isinstance(tensor, Indexed) and tensor.base == IndexedBase('G'):
                G_toUnit = tensor
            elif isinstance(tensor, Indexed) and tensor.base == IndexedBase('H'):
                H_toUnit = tensor
        # Store same-type remainder terms of G and H
        unitTerm = []

        # Start iteration
        if isinstance(add_expr, Add):
            for mul in add_expr.args:
                if G_toUnit in mul.args and H_toUnit in mul.args:
                    unitTerm.append(mul / (G_toUnit * H_toUnit))
                    add_expr -= mul
        elif isinstance(add_expr, Mul):
            # When add_expr has only one term, add_expr becomes Mul class, causing errors
            if G_toUnit in add_expr.args and H_toUnit in add_expr.args:
                unitTerm.append(add_expr / (G_toUnit * H_toUnit))
                add_expr -= add_expr
                    
        united_expr.append(G_toUnit * H_toUnit * Add(*unitTerm))

    return simplify(Add(*united_expr))






