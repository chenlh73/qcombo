'''
输出结果
'''


from IPython.display import display, Latex#latex公式输出
import sympy as sy
sy.init_printing()

from .tools import uniteSimilarTerms


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
        elif(i==',' and str_exp[pos+1]==')'):# do not show ',' when  up or down idx only have one element.
            pass
        else:
            str_lax+=i
        pos+=1
    return str_lax

def jupyterTexDisplay(lat_exp):
    display(Latex(f"$${lat_exp}$$")) 

def _transRightSymbolExpToAmcExp(rightExp):
    str_rightExp=str(rightExp)
    str_amc=''
    for i in str_rightExp:
        if(i=='['):
            str_amc+='_'
        elif(i=='(' or i==')' or i==',' or i==']' or i==' '):
            pass
        elif(i==chr(913)):
            str_amc+='A'
        elif(i==chr(958)):
            str_amc+='xi'
        elif(i==chr(955)):
            str_amc+='lambda'
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

