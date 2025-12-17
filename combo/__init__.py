
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



#TODO
#1. tools.uniteSimilarTerms can't deal with n terms.
#2. uniteSame can't deal with allTerms in 2B2B