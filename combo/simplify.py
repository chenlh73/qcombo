from sympy.combinatorics.permutations import Permutation
from sympy.tensor.indexed import Indexed
from sympy import IndexedBase,symbols,simplify
from sympy.core.mul import Mul
from sympy.core.add import Add
from tqdm import tqdm

#######################################################################
#利用反对称性化简

def sort_indices(indices):
    '''
    对一组指标进行排序，并返回排序后的指标和符号
    
    参数:
        indices: 指标元组
        
    返回:
        (sorted_indices, sign): 排序后的指标和由交换次数决定的符号
    ## Example:
    input: (a,c)  >>  output : (a,c), 1
    input: (b,a)  >>  output : (a,b),-1
    '''
    # 将指标转换为列表
    index_list = list(indices)
    
    # 按字母顺序排序
    sorted_list = sorted(index_list, key=str)
    
    # 计算排列的符号
    if index_list == sorted_list:
        sign = 1
    else:
        # 创建从原始位置到排序后位置的映射
        permutation = []
        for elem in sorted_list:
            permutation.append(index_list.index(elem))
        
        # 计算排列的奇偶性
        p = Permutation(permutation)
        sign = 1 if p.is_even else -1
    
    sorted_tuple = tuple(sorted_list)

    return sorted_tuple, sign

def sort_indexed_tensor(tensor):
    """
    对 Indexed 张量的上下指标分别进行排序，并返回排序后的张量和系数
    
    参数:
        tensor: sympy.tensor.indexed.Indexed 对象
        
    返回:
        (sorted_tensor, coefficient): 排序后的张量和由交换次数决定的系数
    ## Example:
    input: G[(c,a),(e,f)]  >>  output : G[(a,c),(e,f)],-1
    input: G[(c,a),(f,e)]  >>  output : G[(a,c),(e,f)], 1
    """
    if not isinstance(tensor, Indexed):
        raise ValueError("输入必须是 Indexed 对象")
    
    #lambda具有反对称性，因此注释但保留提醒 2025/11/17
    #lambda不具有反对称性, 如果基是 lambda，直接返回原张量和系数 1
    # if tensor.base == IndexedBase(chr(955)):
    #     return tensor, 1
    
    # 获取张量的所有指标组
    index_groups = tensor.indices

    total_sign = 1
    
    # 分离上下指标组
    upper_indices = index_groups[0]
    lower_indices = index_groups[1]

    new_upper_indices,upper_sign = sort_indices(upper_indices)
    new_lower_indices,lower_sign = sort_indices(lower_indices)

    sorted_tensor = tensor.base[new_upper_indices,new_lower_indices]
    total_sign *= upper_sign*lower_sign

    return sorted_tensor, total_sign

def sort_mul_expression(expr):
    """
    对 Mul 表达式中的每个 Indexed 对象进行指标重排
    
    参数:
        expr: sympy.core.mul.Mul 对象
        
    返回:
        sorted_expr: 排序后的表达式
    ## Example:
    input:  G[(c,a),(e,f)]*H[(f,e),(b,a)]   
    >> output: -G[(a,c),(e,f)]*H[(e,f),(a,b)]
    """
    if not isinstance(expr, Mul):
        raise ValueError("输入必须是 Mul 对象")
    
    # 初始化总系数
    total_coefficient = 1
    
    # 存储处理后的因子
    sorted_factors = []
    
    # 处理每个因子
    for tensor in expr.args:
        if isinstance(tensor, Indexed):
            # 获取张量名称
            sorted_tensor, coefficient = sort_indexed_tensor(tensor)
            sorted_factors.append(sorted_tensor)
            total_coefficient *= coefficient
        else:
            # 非 Indexed 对象直接添加
            sorted_factors.append(tensor)
    
    # 构建新的表达式
    sorted_expr = Mul(*sorted_factors)*total_coefficient
    
    return sorted_expr

def sort_add_expression(expr):
    """
    对 Add 表达式中的每一项进行指标重排
    
    参数:
        expr: sympy.core.add.Add 对象
        
    返回:
        sorted_expr: 排序后的表达式
    ## Example:
    input: G[(c,a),(e,f)]*H[(f,e),(b,a)] - G[(a,c),(e,f)]*H[(f,e),(b,a)]  
    >>output: 2*G[(a,c),(e,f)]*H[(e,f),(a,b)]
    """
    if not isinstance(expr, Add):
        raise ValueError("输入必须是 Add 对象")
    
    # 存储处理后的项
    sorted_terms = []
    
    # 处理每一项
    with tqdm(total=len(expr.args), desc="simplifying") as pbar:
        for term in expr.args:
            if isinstance(term, Mul):
                # 处理 Mul 项
                sorted_term= sort_mul_expression(term)
                sorted_terms.append(sorted_term)
            elif isinstance(term, Indexed):
                # 处理单个 Indexed 项
                sorted_term, sign = sort_indexed_tensor(term)
                sorted_terms.append(sorted_term*sign)
            else:
                # 其他类型的项直接添加
                sorted_terms.append(term)
            pbar.update(1)
    
    # 构建新的表达式
    sorted_expr = Add(*sorted_terms)
    
    return sorted_expr

#######################################################################
#利用傀儡指标的性质化简含有lamdda项的表达式

def reorder_dummy_indices_mul(expr):
    """
    重新排列傀儡指标,使得λ的指标变成除A指标外按字母表排序的前几个指标
    
    参数:
    expr: Mul表达式
    
    返回:
    重新标记后的表达式
    """
    if not isinstance(expr, Mul):
        raise ValueError("输入必须是 Mul 对象")
    
    # 提取表达式中的各个张量
    factors = expr.args
    
    A_tensor = None
    lambda_tensor = []  #对于多个lambda变量同样可以实现

    for tensor in factors:
        if isinstance(tensor, Indexed): #保证是张量对象
            if tensor.base == IndexedBase('A'):
                A_tensor = tensor
            elif tensor.base == IndexedBase(chr(955)):
                lambda_tensor.append(tensor)

    #如果没有lambda项直接返回原式
    if len(lambda_tensor) == 0:
        return expr

    # 获取A的指标（这些是固定指标，不参与重新标记）
    A_indices = set()
    if A_tensor != None: #对于零体项将没有A项
        for index_tuple in A_tensor.indices:    
            A_indices.update(index_tuple)
    
    # 收集所有傀儡指标
    all_dummy_indices = set()
    for tensor in factors:
        if isinstance(tensor, Indexed): #保证是张量对象
            if tensor.base != IndexedBase('A'):
                for index_tuple in tensor.indices:
                    all_dummy_indices.update(index_tuple)
    
    # 移除A的指标(原则上A并不在这里)
    dummy_indices = all_dummy_indices - A_indices
    
    # 按字母表排序傀儡指标
    sorted_dummy_indices = sorted(dummy_indices, key=str)
    
    # 获取lambda的初始指标
    lambda_flat_indices = []
    for tensor in lambda_tensor:
        for index_tuple in tensor.indices:
            lambda_flat_indices.extend(index_tuple)

    # 确定λ需要多少个新指标
    lambda_index_count = len(lambda_flat_indices) 

    # 为λ分配排序后的前几个指标
    new_lambda_indices = sorted_dummy_indices[:lambda_index_count]
    
    # 为剩余的傀儡指标分配排序后的其他指标
    remaining_indices = sorted_dummy_indices[lambda_index_count:]

    # 创建指标映射
    index_mapping = {}
    
    # 首先映射λ的指标
    for old_idx, new_idx in zip(lambda_flat_indices, new_lambda_indices):
        index_mapping[old_idx] = new_idx
    
    # 然后映射剩余的傀儡指标
    #remaining_old_indices = list(dummy_indices - set(lambda_flat_indices))
    remaining_old_indices = sorted(dummy_indices - set(lambda_flat_indices),key=str) #剩余的傀儡指标再次排列,update:2025/11/18
    for old_idx, new_idx in zip(remaining_old_indices, remaining_indices):
        index_mapping[old_idx] = new_idx
    
    # 应用指标映射到各个张量
    new_expr = expr.xreplace(index_mapping)
    
    return new_expr


def reorder_dummy_indices_add(expr):
    """
    重新排列傀儡指标,使得λ的指标变成除A指标外按字母表排序的前几个指标
    
    参数:
    expr: Add表达式
    
    返回:
    重新标记后的表达式
    """
    if not isinstance(expr, Add):
        raise ValueError("输入必须是 Add 对象")
    
    # 存储处理后的项
    reorder_terms = []
    
    # 处理每一项
    with tqdm(total=len(expr.args), desc="simplifying") as pbar:
        for term in expr.args:
            reorder_terms.append(reorder_dummy_indices_mul(term))
            pbar.update(1)
            
    # 构建新的表达式
    reorder_expr = Add(*reorder_terms)

    return reorder_expr

#######################################################
#根据含有的lambda多项式的不同，对表达式进行筛选
# add by chenlh, 2025/11/12

def get_Indexed_IndicesNum(tensor):
    """
    获取张量的指标个数（阶数）
    参数:
        tensor: 张量对象，可以是 Indexed 对象或其他包含指标的对象
        
    返回:
        张量的指标个数（整数）
    """
    # 检查是否是 SymPy Indexed 对象
    if not isinstance(tensor, Indexed):
        raise ValueError("输入必须是 Indexed 对象")
    
    return len(tensor.indices[0]) + len(tensor.indices[1])

def filterLambdaBody(expr,filterLambdaBody):
    """
    根据 Lambda 张量的指标个数筛选表达式中的项
    
    参数:
        expr: SymPy 表达式 (Add 类)
        filterLambdaBody: 目标 Lambda 张量的多体项,整数
    返回:
        筛选后的表达式
    # Example
    expr = A[(a,b),(c,d)]*lambda[(e),(f)] + A[(a,b),(c,d)]*lambda[(e,f),(g,h)]
    
    filterLambdaBody = 2

    >> A[(a,b),(c,d)]*lambda[(e,f),(g,h)]
    """
    # 确保表达式是 Add 类型
    if not isinstance(expr, Add):
        raise ValueError("表达式必须是 Add 类型")
    
    # 筛选符合条件的项
    filtered_terms = []
    no_lambda_tetms = []

    for mul in expr.args:  # 遍历Add类中每一项mul类
        lambda_found = False
        lambda_body = 0
        for tensor in mul.args:   # 遍历mul类中每一项indexed类
            if isinstance(tensor, Indexed):
                if tensor.base == IndexedBase(chr(955)):
                    lambda_found = True
                    lambda_body = int((get_Indexed_IndicesNum(tensor))/2)
                    if lambda_body == filterLambdaBody:
                        filtered_terms.append(mul)
                    break
        #如果找不到lambda，将加入到no_lambda_tetms 
        if not lambda_found:
            no_lambda_tetms.append(mul)

    #如果筛选的lambda多体项为0或1，则返回没有lambda的项
    if filterLambdaBody == 0 or filterLambdaBody ==1:
        return Add(*no_lambda_tetms) 
    #否则正常返回
    return Add(*filtered_terms)

#合并具有相同的G和H的项
#add by chenlh 2025/11/17
#update:2025/11/20 修复了当表达式中没有可化简的项时就会持续运行的bug

def uniteSameGAndH(expr):

    #--
    if not isinstance(expr, Add):
        raise ValueError("输入必须是 Add 对象")
    
    united_expr = []

    add_expr = expr

    #！！！！！！！！！！！！！！！
    #当add_expr只有一项的时候，add_expr变为mul类，从而导致报错

    while add_expr != 0 :
        #取第一项的G和H进行化简
        #当add_expr只有一项的时候，add_expr变为mul类，从而导致报错
        if isinstance(add_expr,Add):
            first_mul_term = add_expr.args[0]
        elif isinstance(add_expr,Mul):
            first_mul_term = add_expr
        #找到待合并的G和H
        for tensor in first_mul_term.args:
            if isinstance(tensor,Indexed) and tensor.base == IndexedBase('G'):
                G_toUnit = tensor
            elif isinstance(tensor,Indexed) and tensor.base == IndexedBase('H'):
                H_toUnit = tensor
        #储存G和H的同类余项
        unitTerm = []

        #开始遍历
        if isinstance(add_expr,Add):
            for mul in add_expr.args:
                if G_toUnit in mul.args and H_toUnit in mul.args:
                    unitTerm.append(mul/(G_toUnit*H_toUnit))
                    add_expr -=  mul
        elif isinstance(add_expr,Mul):
        #当add_expr只有一项的时候，add_expr变为mul类，从而导致报错
            if G_toUnit in add_expr.args and H_toUnit in add_expr.args:
                    unitTerm.append(add_expr/(G_toUnit*H_toUnit))
                    add_expr -=  add_expr
                    
        united_expr.append(G_toUnit*H_toUnit*Add(*unitTerm))

    return simplify(Add(*united_expr))







