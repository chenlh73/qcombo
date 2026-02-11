from IPython.display import display, Latex#latex output
import sympy as sy
from sympy import preorder_traversal,IndexedBase,factorial,Rational
from sympy.tensor.indexed import Indexed
sy.init_printing()
from .tools import get_all_indices


def transSymbolsToLatex(tmp):
    #TODO :
    #Maybe I can use stack to achieve this function
    str_exp=str(tmp)
    str_lax=''
    state_body=0#1  means process a multi terms 
    state_idx=0# =1 finish processing  the up idx,=2 means finish processing the dow idx.
    pos=0# store the work position of str_exp
    for i in str_exp:
        if(state_body==0 and i=='['):
            str_lax+='^'
            state_body=1
        elif(state_body==1 and i=='('):
            str_lax+='{'
        elif(state_body==1 and state_idx==0 and i==')'):
            str_lax+='}'
            state_idx=1
        elif(state_idx==1 and i==','):
            str_lax+='_'
            state_idx=2
        elif(state_body==1 and state_idx==2 and i==')'):
            str_lax+='}'
            state_idx=0
        elif(state_body==1 and i==']'):
            state_body=0
        elif(i=='*'):
            pass
        elif(i==chr(913)):
            str_lax+='A'
        elif(i==chr(958)):
            str_lax+='\\xi'
        elif(i==chr(955)):
            str_lax+='\\lambda'
        elif(i==chr(948)):
            str_lax+='\\delta'
        # elif(i==',' and str_exp[pos+1]==')'):# do not show ',' when  up or down idx only have one element.
        #     pass
        elif(i==',' ):# do not show ',' in anytime, different with above  modified by L.H.Chen 2025/12/23
            pass            
        else:
            str_lax+=i
        pos+=1
    return str_lax

def jupyterTexDisplay(lat_exp):
    display(Latex(f"$${lat_exp}$$")) 

# modified by L.H.Chen 2025/12/23 
def _transRightSymbolExpToAmcExp(rightExp,lambdaStr='lambda'):
    str_rightExp=str(rightExp)
    strInIndices = False
    str_amc=''
    for i in str_rightExp:
        if(i=='['):
            str_amc+='_'
            strInIndices = True
        # elif(i=='(' or i==')' or i==',' or i==']' or i==' '):
        #     pass
        elif ((i=='(' or i ==')' ) and strInIndices):  #省略指标内的括号，但保留指标外的括号
            pass
        elif (i==']'):
            strInIndices = False
            pass
        elif( i==',' or i==' '):
            pass
        elif(i==chr(913)):
            str_amc+='A'
        elif(i==chr(958)):
            str_amc+='xi'
        elif(i==chr(955)):
            #str_amc+='lambda'
            str_amc += lambdaStr #区分两体或三体lambda
        elif(i==chr(948)):
            str_amc+='delta'
        else:
            str_amc+=i
    return str_amc

def _extractIndicesFromTuple(indicesTuple1):
    res=""
    for i in indicesTuple1:
        res+=str(i)
    return res

def _transSymbolExpToAmcExp(exp,indices):
    retainIndices=()# Can I remoove it?
    expExpand=exp.expand()
    aftreRemoveAExp=0
    for i in expExpand.args:
        tmp=1
        for j in i.args:
            if(str(j)[0]!='A'):
                tmp*=j
            else:#get retain indices from A trems.
                retainIndices=j.args[1]+j.args[2]
        aftreRemoveAExp+=tmp
    # rightTerms=uniteSimilarTerms(aftreRemoveAExp)
    # Form left term of amc equation
    leftSub=_extractIndicesFromTuple(retainIndices)
    # Form right terms of amc equation
    sumIndices=[]
    for i in indices:
        if i not in retainIndices:
            sumIndices.append(i)
    amcRight='1/4*sum_'+_extractIndicesFromTuple(sumIndices)+'('+_transRightSymbolExpToAmcExp(aftreRemoveAExp)+');'

    return leftSub, amcRight

def _getLen(exp):
    exp=exp.expand()#exp: G[]*H[]-H[]lambda[]
    lenSet={chr(955):0}
    for i in exp.args:
        for j in i.args:
            if(j!=-1):
                lenSet[str(j.args[0])]=( len(j.args[1]), len(j.args[2]) )
    return lenSet

def amcInputFIle(exp,indices):
    '''
    Form the amc input file using the rule of amc.
    
    '''
    # amc left and right
    leftSub,amcRight=_transSymbolExpToAmcExp(exp,indices)
    #declare
    lenSet=_getLen(exp)
    lenR=len(leftSub)
    lenG =lenSet['G']
    lenH =lenSet['H']
    lenLambda =lenSet[chr(955)]
    declareR='declare R { ' + f'mode= {lenR},' + 'latex ="R"} \n'
    declareG='declare G { ' + f'mode= {lenG},' + 'latex ="G"} \n'
    declareH='declare H { ' + f'mode= {lenH},' +'latex="H" }\n'
    declareLambda='declare lambda { ' + f'mode= {lenLambda},' + 'latex="\lambda" } \n'
    declare_n='declare n {  mode=2, diagonal=true, latex="n"} \n'
    #Equation
    if(lenR==0):
        amcLeft='R'
    else:
        amcLeft='R_'+leftSub
    equation=amcLeft+'='+amcRight
    #amc document
    amctxt=declareR+ declareG+declareH+declareLambda+declare_n+equation
    with open('./output.amc', 'w', encoding='utf-8') as f:
        f.write(amctxt)
        print('\nSave output.amc successfully!\n' )



###################################################
#add by L.H.Chen 2025/12/23
def sympyExprToLatex(expr):
    #TODO :
    #Maybe I can use stack to achieve this function
    str_exp=str(expr)
    str_lax=''
    state_body=0#1  means process a multi terms 
    state_idx=0# =1 finish processing  the up idx,=2 means finish processing the dow idx.
    pos=0# store the work position of str_exp
    for i in str_exp:
        if(state_body==0 and i=='['):
            str_lax+='^'
            state_body=1
        elif(state_body==1 and i=='('):
            str_lax+='{'
        elif(state_body==1 and state_idx==0 and i==')'):
            str_lax+='}'
            state_idx=1
        elif(state_idx==1 and i==','):
            str_lax+='_'
            state_idx=2
        elif(state_body==1 and state_idx==2 and i==')'):
            str_lax+='}'
            state_idx=0
        elif(state_body==1 and i==']'):
            state_body=0
        elif(i=='*'):
            pass
        elif(i==chr(913)):
            str_lax+='A'
        elif(i==chr(958)):
            str_lax+='\\xi'
        elif(i==chr(955)):
            str_lax+='\\lambda'
        elif(i==chr(948)):
            str_lax+='\\delta'
        elif(i==',' ):# do not show ',' 
            pass
        else:
            str_lax+=i
        pos+=1
    return str_lax

def LatexDisplay(latex_str):
    display(Latex(f"$${sympyExprToLatex(latex_str)}$$"))



################################################################
# the Function for easycCombo

def transExprToEquationLatex(tmp):
    r'''
    retrun the latex string for equation, the * trans to \ *
    '''
    str_exp=str(tmp)
    str_lax=''
    state_body=0#1  means process a multi terms 
    state_idx=0# =1 finish processing  the up idx,=2 means finish processing the dow idx.
    pos=0# store the work position of str_exp
    for i in str_exp:
        if(state_body==0 and i=='['):
            str_lax+='^'
            state_body=1
        elif(state_body==1 and i=='('):
            str_lax+='{'
        elif(state_body==1 and state_idx==0 and i==')'):
            str_lax+='}'
            state_idx=1
        elif(state_idx==1 and i==','):
            str_lax+='_'
            state_idx=2
        elif(state_body==1 and state_idx==2 and i==')'):
            str_lax+='}'
            state_idx=0
        elif(state_body==1 and i==']'):
            state_body=0
        elif(i=='*'):
            str_lax+= ' \\* '
        elif(i==chr(913)):
            str_lax+='A'
        elif(i==chr(958)):
            str_lax+='\\xi'
        elif(i==chr(955)):
            str_lax+='\\lambda'
        elif(i==chr(948)):
            str_lax+='\\delta'
        # elif(i==',' and str_exp[pos+1]==')'):# do not show ',' when  up or down idx only have one element.
        #     pass
        elif(i==',' ):# do not show ',' in anytime, different with above  modified by L.H.Chen 2025/12/23
            pass
        elif(i == ' '): # don't show space' ', more tighter     
            pass        
        else:
            str_lax+=i
        pos+=1
    return str_lax


def getEquationLatexStrFromExpr(expr, left_body, right_body, contractionBase='R'):
    """
    Convert a SymPy expression to a LaTeX equation string
    e.g., G[(a),(c)] H[(c),(b)] A[(a),(b)] -> R^{a}_{b} = sum_{c} G^{a}_{c} H^{c}_{b} 
    """
    # Find the A operator
    A = IndexedBase('A')
    A_tensor = 1

    for term in preorder_traversal(expr):
        if isinstance(term, Indexed):
            if term.base == A:
                A_tensor = term

    # Create contraction operator base
    if isinstance(contractionBase, str):
        contraction_base = IndexedBase(contractionBase)  
    elif isinstance(contractionBase, IndexedBase):
        contraction_base = contractionBase

    if A_tensor == 1:  # Expression contracts to zero-body term
        contraction_tensor = contraction_base[(),()]
    else:
        contraction_tensor = contraction_base[A_tensor.indices]
    
    contraction_upIndices, contraction_downIndices = contraction_tensor.indices

    coefficient_1 = factorial(len(contraction_upIndices)) * factorial(len(contraction_downIndices))
    coefficient_2 = (factorial(left_body)**2) * (factorial(right_body)**2)
    Right_Equation = coefficient_1 * (1/coefficient_2) * expr / A_tensor
    Left_Equation = contraction_tensor

    # Get summation indices
    all_indices = get_all_indices(Right_Equation)
    contraction_indices = get_all_indices(Left_Equation)
    dummy_indices = set(all_indices) - set(contraction_indices)
    dummy_indices = sorted(dummy_indices, key=str)

    # Add summation LaTeX output
    sum_Latex_str = r'\sum_{'
    for i in dummy_indices:
        sum_Latex_str += f'{i}'
    sum_Latex_str += '} '

    # Create LaTeX expressions for left and right sides
    Left_Latex_str = transExprToEquationLatex(Left_Equation)
    Right_Latex_str = transExprToEquationLatex(Right_Equation)

    return Left_Latex_str + ' = ' + sum_Latex_str + Right_Latex_str


def outputLatexStr(expr_dict, filter_body_list, left_body, right_body, contractionBase='R'):
    '''
    expr_dict: dictionary, e.g. {'2B_lambda2B': sympy expr}
    filter_body_list: list, the contraction body

    Returns
    -------
    latex : str, A LaTeX document as a string.
    '''
    # Write LaTeX document header
    output = [r'''
    \documentclass{scrartcl}

    \usepackage{expl3}
    \usepackage{xparse}
    \usepackage{amsmath}
    \usepackage{breqn}

    \newcommand{\threej}[3]{\begingroup\setlength{\arraycolsep}{0.2em}\begin{Bmatrix} #1 & #2 & #3 \end{Bmatrix}\endgroup}
    \newcommand{\sixj}[6]{\begingroup\setlength{\arraycolsep}{0.2em}\begin{Bmatrix} #1 & #2 & #3 \\ #4 & #5 & #6 \end{Bmatrix}\endgroup}
    \newcommand{\ninej}[9]{\begingroup\setlength{\arraycolsep}{0.2em}\begin{Bmatrix} #1 & #2 & #3 \\ #4 & #5 & #6 \\ #7 & #8 & #9 \end{Bmatrix}\endgroup}

    \ExplSyntaxOn
    \tl_new:N \l__hatfact_tl

    \NewDocumentCommand{\hatfact} {m} { \__hatfact_parse:n #1 }

    \cs_new:Nn \__hatfact_parse:n { \tl_set:Nn \l__hatfact_main_tl {#1} \__hatfact_hat: }

    \cs_generate_variant:Nn \str_set:Nn {Nx}
    \cs_new:Nn \__hatfact_hat:
    {
        \str_set:Nx \l_tmpa_str {\l__hatfact_main_tl}
        \str_set:Nn \l_tmpb_str {j}
        \str_if_eq:NNTF \l_tmpa_str \l_tmpb_str {
        \hat{\jmath}
        } {
        \hat \l__hatfact_main_tl
        }
    }
    \ExplSyntaxOff

    \begin{document}
    ''']
    max_contraction_body = left_body + right_body
    for filter_body in filter_body_list:
        output.append(r'\section{{Equation [{}B,{}B]-{}B}}'.format(left_body, right_body, filter_body))
        for lambda_body in range(1, max_contraction_body):
            try:
                united_expr = expr_dict[f'{filter_body}B_lambda{lambda_body}B']
            except KeyError:
                # If KeyError, lambda_body does not exist, skip to next
                continue
            else:
                # Start writing
                output.append(r'\subsection{{$\lambda^{{{}B}}$}}'.format(lambda_body))
                output.append(r'\begin{dmath*}')
                output.append(getEquationLatexStrFromExpr(united_expr, left_body, right_body, contractionBase))
                output.append(r'\end{dmath*}')

    output.append(r'\end{document}')
    output.append(r'')

    return '\n'.join(output)

def getEquationAmcStrFromExpr(expr, left_body, right_body, contractionBase_str):
    """
    Convert SymPy expression to AMC equation string
    """
    # Find A and lambda operators
    A = IndexedBase('A')
    A_tensor = 1
    lambda_base = IndexedBase(chr(955))  # Lambda symbol
    lambda_tensor = None

    for term in preorder_traversal(expr):
        if isinstance(term, Indexed):
            if term.base == A:
                A_tensor = term
            elif term.base == lambda_base:
                lambda_tensor = term
    
    # Build contracted expression operator
    contraction_base = IndexedBase('R')  # Create contraction operator base
    if A_tensor == 1:  # Expression contracts to zero-body term
        contraction_body = 0
        contraction_base = IndexedBase(contractionBase_str + f'{contraction_body}') 
        contraction_tensor = contraction_base[(),()]
    else:
        contraction_body = len(A_tensor.indices[1])
        contraction_base = IndexedBase(contractionBase_str + f'{contraction_body}') 
        contraction_tensor = contraction_base[A_tensor.indices]

    contraction_upIndices, contraction_downIndices = contraction_tensor.indices

    coefficient_1 = factorial(len(contraction_upIndices)) * factorial(len(contraction_downIndices))
    coefficient_2 = (factorial(left_body)**2) * (factorial(right_body)**2)
    coefficient = Rational(coefficient_1, coefficient_2)
    Right_Equation = expr / A_tensor
    Left_Equation = contraction_tensor

    # Get summation indices
    all_indices = get_all_indices(Right_Equation)
    contraction_indices = get_all_indices(Left_Equation)
    dummy_indices = set(all_indices) - set(contraction_indices)
    dummy_indices = sorted(dummy_indices, key=str)

    # Add summation AMC output
    sum_amc_str = 'sum_'
    for i in dummy_indices:
        sum_amc_str += f'{i}'

    # Create AMC expressions for left and right sides
    Left_amc_str = _transRightSymbolExpToAmcExp(Left_Equation)
    
    if A_tensor == 1:  # Contract to zero-body term
        Left_amc_str = f'{contraction_base}'  # Will not display indices
    
    # Input different lambda string based on whether lambda is two-body or three-body term
    if lambda_tensor is not None:
        lambda_body = len(lambda_tensor.indices[1])
        Right_amc_str = _transRightSymbolExpToAmcExp(Right_Equation, lambdaStr=f'lambda{lambda_body}B')
    else:
        Right_amc_str = _transRightSymbolExpToAmcExp(Right_Equation)

    return Left_amc_str + ' = ' + str(coefficient) + '*' + sum_amc_str + '(' + Right_amc_str + ');'


def outputAmcStr(expr_dict, contraction_list, max_lambda_body, left_body, right_body, leftBase_str, rightBase_str, contractionBase_str):
    '''
    expr_dict: dictionary, e.g. {'2B_lambda2B': sympy expr}
    contraction_list: list, the contraction body
    
    Returns
    -------
    amc : str, An AMC document as a string.
    '''
    output = []
    # Declare operators
    declareLeft = f'declare {leftBase_str}' + '{' + f'mode= ({left_body},{left_body}),' + f'latex ="{leftBase_str}" ' + '} ' 
    output.append(declareLeft)
    declareRight = f'declare {rightBase_str}' + '{' + f'mode= ({right_body},{right_body}),' + f'latex ="{rightBase_str}" ' + '} ' 
    output.append(declareRight)

    for contraction_body in contraction_list:
        declareContraction = f'declare {contractionBase_str}{contraction_body}' + '{' + f'mode= ({contraction_body},{contraction_body}),' + f'latex ="{contractionBase_str}" ' + '} ' 
        output.append(declareContraction)

    for lambda_body in range(2, max_lambda_body + 1):
        declareLambda = f'declare lambda{lambda_body}B' + '{' + f'mode= ({lambda_body},{lambda_body}),' + r'latex ="\lambda" ' + '} ' 
        output.append(declareLambda)

    declare_n = 'declare n {  mode=2, diagonal=true, latex="n"} '
    output.append(declare_n)

    output.append(' ')

    # Input AMC formulas
    max_contraction_body = left_body + right_body
    for filter_body in contraction_list:
        output.append(r'# commutator [{}B,{}B]-{}B '.format(left_body, right_body, filter_body))
        for lambda_body in range(1, max_contraction_body):
            try:
                united_expr = expr_dict[f'{filter_body}B_lambda{lambda_body}B']
            except KeyError:
                # If KeyError, lambda_body does not exist, skip to next
                continue
            else:
                # Start writing
                output.append(r'# lambda_{}B'.format(lambda_body))
                output.append(getEquationAmcStrFromExpr(united_expr, left_body, right_body, contractionBase_str))
                output.append(' ')

    output.append(r'')

    return '\n'.join(output)





