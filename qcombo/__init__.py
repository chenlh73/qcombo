# A package for computing the commutator of many body operator 
# based on generalized Wick's theorm in quantum mechanics.
#
# Copyright (C) 2025 L.H.chen, Y.Li, Heiko Hergert, J.M.Yao
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

__license__ = 'MIT'
__author__ = 'L.H.chen, Y.Li, Heiko Hergert, J.M.Yao'
__version__ = '0.1.4'


from sympy import IndexedBase,symbols
from .wickcaculate import Wick 
from . import canonical
from .tools import SimplifyRule,Filter,uniteSimilarTerms,indicesMultToSimp
from . import output
from .simplify import sort_add_expression,reorder_dummy_indices_add,filterLambdaBody,uniteSameGAndH

A=IndexedBase('A')
G=IndexedBase('G')
H=IndexedBase('H')
delta=IndexedBase(chr(948))



class bodys:
    def __init__(self, leftIndices, rightIndices) -> None:
        leftIndices=[list(map(lambda x:symbols(x),i))for i in leftIndices]
        rightIndices=[list(map(lambda x:symbols(x),i))for i in rightIndices]
        self.indices=[leftIndices,rightIndices]

    def help(self):
        '''
        Display the meaning of attributes.
        '''
        print("# The attribute in class bodys")
        print("Attribute".rjust(15),"Description")
        atrMsg=[
            ["gw","Generalized Wick Result"],
            ["cmt", "gw1 - gw2"],
            ["cmtRule","The result after applying xiRule or natRule (or both)to cmt"],
            ["allTerms","All Terms after applying  xiRule and natRule"],
            ["filterTerms","Filter Result"],
            ["canon","Canonicalized Result(Deal the deal term and rename the indices)"], 
        ]
        for i in atrMsg:
            print((i[0]+':').rjust(15),i[1])

        print("\n\n")
        print("# The methods in class bodys")
        print("Method".rjust(15),"Description")
        mtdMsg=[
            ["commutate","Caculate commutation relation."],
            ["apply","Apply rule to the commutation relation result."],
            ["regular","Filter your input body type and  Canonicalized the filter terms(deal the deal term and rename the indices )"],
            ["amcExport","Form a amc document based on the regular result(after filtering)"],
            ["amcEnd","Do the above 4 steps  in this function directly."],
            ["texplay","Display all step result by latex in jupyter. "],

        ]
        for i in mtdMsg:
            print((i[0]+':').rjust(15),i[1])


        print("\n\n")
        print("# The function in mboc")
        print("Function".rjust(15),"Description")
        fucMsg=[
            ["texExp","Trans the expression to latex."],
            ["uniteSame","Combine the same terms."],
            ["jupyterDisplay","Display the expression by latex form in jupyter"],
        ]
        for i in fucMsg:
            print((i[0]+':').rjust(15),i[1])

    def texplay(self,jupyter="yes"):
        '''
        Display all step result by latex in jupyter.If you just want to show the latex expression, please set  the parameter: jupyter="no".
        '''
        attributes=["gw","cmt","cmtRule","allTerms","filterTerms","canon",]
        if jupyter=="yes":
            for i in attributes:
                try:
                    exp=getattr(self,i)
                    laxExp=output.transSymbolsToLatex(exp)
                    if i=="filterTerms":
                       print( "filterTerms("+ str(filterBodyType) + "bodys) :")
                    else:
                        print(i,":") 
                    output.jupyterTexDisplay(laxExp)
                except:
                    print('No attribute called:',i )
        if jupyter=="no":
            for i in attributes:
                try:
                    exp=getattr(self,i)
                    laxExp=output.transSymbolsToLatex(exp)
                    if i=="filterTerms":
                       print( "filterTerms("+ str(filterBodyType) + "bodys) :")
                    else:
                        print(i,":") 
                    print(laxExp) 
                except:
                    print('No attribute called:',i )

    def commutate(self):
        wickCase=Wick(self.indices[0],self.indices[1])
        wickCase.commmutate()
        self.gw=wickCase.gw
        self.cmt=wickCase.cmt
    #Infact it is not the commutate result.

    def applyRule(self,ruleType='both'):
        '''
        Chose your rule that apply to commutate result. The default value of ruleType is 'both', and you can set 'xi' or 'nat' instead.
        '''
        if ruleType=='both':
            self.cmtRule=SimplifyRule.natRule(SimplifyRule.xiRule(self.cmt))
        elif ruleType=='xi':
            self.cmtRule=SimplifyRule.xiRule(self.cmt)
        elif ruleType=='nat':
            self.cmtRule=SimplifyRule.natRule(self.cmt)
   
    def regulate(self,filterbody=None):
        """
        
        """
        self.allTerms=(G[tuple(self.indices[0][0]),tuple(self.indices[0][1])]*H[tuple(self.indices[1][0]),tuple(self.indices[1][1])]*self.cmtRule).expand()
        #Fliter
        if filterbody!=None and type(filterbody)==int:
            global filterBodyType
            filterBodyType=filterbody
            self.filterTerms=Filter.filterbody(self.allTerms,filterbody)
        else:
            self.filterTerms=self.allTerms
        
        #canonicalize
        canonTemp=canonical.canonicalize(self.filterTerms)
        self.canon,self.indicesSet=indicesMultToSimp(canonTemp)
    
    def amcExport(self):
        output.amcInputFIle(self.canon,self.indicesSet)

    def amcEnd(self,applyRule="both",fliterbody=None):
        self.commutate()
        print('Finish commutate!')
        self.applyRule(applyRule)
        print('Finish apply Rule!')
        self.regulate(fliterbody)
        print('Finish filter'+str(fliterbody)+'body,and regulate it!')
        print('Start form amc document!')
        self.amcExport()

def texExp(exp):
    lat_exp=output.transSymbolsToLatex(exp)
    return lat_exp
def filterbody(exp,bodyType):
    '''
    Filter input bodyType from exp.
    '''
    return Filter.filterbody(exp,bodyType)

def jupyterDisplay(exp):
    output.jupyterTexDisplay(texExp(exp))

def uniteSame(exp):
    return uniteSimilarTerms(exp.expand())

#simplify, add by lhchen 2025/11/11
def simplifyUseAntisymmetry(expr):
    return sort_add_expression(expr)

def simplifyUseDummyIndices(expr):
    return reorder_dummy_indices_add(expr)

def simplifyUseBoth(expr):
    sorted_expr = sort_add_expression(expr)
    reordered_expr = reorder_dummy_indices_add(sorted_expr)
    return sort_add_expression(reordered_expr)

#####################################################################
#
def easyCombo(left, right, contraction=None, latexOutput=None, amcOutput=None):
    '''
    Calculate the commutator [left, right] to specified contraction, e.g., easyCombo(2, 2)

    ## Parameters
    left: int or string list, the body or indices of left operator e.g., 1 or ['a','b'], they are equivalent
    right: int or string list, the right operator, same as left
    contraction: int or int list, the body of commutator to obtain e.g., 0 or [0,1,...]
    latexOutput: the file name of the output tex file
    amcOutput: the file name of the amc input file

    ## Warning
    Ensure contraction terms are valid,
    Input left and right indices must not be identical
    '''
    
    # Predefined index list
    indiceListPre = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                     'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    # Check if input indices are string lists; if not, automatically assign from predefined list
    if isinstance(left, list) and isinstance(right, list):
        left_indices = left
        right_indices = right
        max_contraction_body = len(left_indices[0]) + len(right_indices[0])
    elif isinstance(left, int) and isinstance(right, int):
        left_indices = [indiceListPre[:left], indiceListPre[left:left*2]]
        right_indices = [indiceListPre[left*2:left*2+right], indiceListPre[left*2+right:left*2+right*2]] 
        max_contraction_body = left + right 
    else:
        raise ValueError("Input must be both integers or both string lists")
    
    # Determine contraction term type
    if contraction is None:
        contraction_list = [i for i in range(0, max_contraction_body)]  # Auto-generate list of maximum attainable couplings
    else:
        if isinstance(contraction, list):
            contraction_list = contraction
        elif isinstance(contraction, int):
            contraction_list = list([contraction])
        # Validate contraction range
        for i in contraction_list:
            if i >= max_contraction_body or i < 0:
                raise ValueError(f'Contraction term [{i}] exceeds commutator range!')
        
    #######################################################################
    # Computation section
    commutator = bodys(left_indices, right_indices)

    # Calculate commutator
    commutator.commutate()
    # Diagonalize single-particle density matrix
    commutator.applyRule()
    
    # Create dictionary to store expressions
    expr_dict = {}
    max_lambda_body = 1

    # Perform contractions for each specified contraction term
    for filter_body in contraction_list:
        print(f'Contracting to {filter_body}-body terms')
        # Filter terms coupling to k-body
        commutator.regulate(filter_body)

        # Get canonical (but not simplified) expression
        initial_expr = commutator.canon

        # Simplify the initial expression
        simplified_expr = simplifyUseBoth(initial_expr)

        # Classify and filter by lambda terms:
        for lambda_body in range(1, max_contraction_body):  # Assume no density matrix exceeds maximum contraction
            filter_lambda_expr = filterLambdaBody(simplified_expr, lambda_body)
            # If filtered lambda expression is non-zero, merge G and H with same indices
            if filter_lambda_expr != 0:
                united_expr = uniteSameGAndH(filter_lambda_expr)
                # Future: add matrix element operator replacement functionality here
                G = IndexedBase('G')  # Original left operator base
                H = IndexedBase('H')  # Original right operator base
                left_Base = IndexedBase('G')  # New left operator base
                right_Base = IndexedBase('H')  # New right operator base
                united_expr = united_expr.xreplace({G: left_Base, H: right_Base})
                # Store in dictionary with key like '2B_lambda2B'
                expr_dict[f'{filter_body}B_lambda{lambda_body}B'] = united_expr
                if max_lambda_body < lambda_body:
                    max_lambda_body = lambda_body

    print("\n" + "=" * 50)
    print("Calculation completed!")

    # Define left, right, and contraction operator bases
    left_body = len(left_indices[0])
    right_body = len(right_indices[0])
    left_Base_str = f'{left_Base}'
    right_Base_str = f'{right_Base}'
    contraction_Base_str = 'R'
    
    ##########################################################
    # LaTeX output
    # LaTeX file naming
    # e.g., commutator_2B2B_to_0_1_2B.tex
    if latexOutput is None:
        latexOutput = f'commutator_{left_body}B{right_body}B_to'
        for i in contraction_list:
            latexOutput += f'_{i}'
        latexOutput += 'B.tex'

    # LaTeX output
    output_tex = output.outputLatexStr(expr_dict, contraction_list, left_body, right_body, contraction_Base_str)

    with open(latexOutput, 'wt') as f:
        f.write(output_tex)

    print(f"LaTeX output saved to: {latexOutput}")

    ###########################################################
    # AMC program input file output
    # AMC file naming
    # e.g., commutator_2B2B_to_0_1_2B_amcInput.txt
    if amcOutput is None:
        amcOutput = f'commutator_{left_body}B{right_body}B_to'
        for i in contraction_list:
            amcOutput += f'_{i}'
        amcOutput += 'B_amcInput.amc'
    

    # AMC output
    output_amc = output.outputAmcStr(expr_dict, contraction_list, max_lambda_body, left_body, right_body,
                                     left_Base_str, right_Base_str, contraction_Base_str)  
    
    with open(amcOutput, 'w') as f:
        f.write(output_amc)

    print(f"AMC input file saved to: {amcOutput}")
    
    return expr_dict



#TODO
# 1.can't deal with the particle number changed operator
# 2.simplify function still unperfect