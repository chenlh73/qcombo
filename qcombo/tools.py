from sympy import IndexedBase#用于存储指标
from sympy import simplify
from sympy import symbols,preorder_traversal
from sympy.tensor.indexed import Indexed
import time


#show progress  update by Chen.L.H 2025/12/22 
class ProgressBar:
    def __init__(self, total, description="处理中", bar_length=50):
        self.total = total
        self.description = description
        self.bar_length = bar_length
        self.current = 0
        self.start_time = time.time()
        
    def update(self):
        self.current += 1
        progress = self.current / self.total
        progress_info = f"[{self.current}/{self.total}]{progress:.1%}"
        
        # 进度条
        filled_length = int(self.bar_length * progress)
        bar = '█' * filled_length + '░' * (self.bar_length - filled_length)
        
        # 时间估计
        elapsed = time.time() - self.start_time
        if self.current > 1:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            time_info = f" | Remaining: {remaining:.1f}s"
        else:
            time_info = ""
        
        print(f'\r{self.description}: [{bar}] {progress_info} {time_info}', end='')
        

        if self.current == self.total:
            print(f'\n{self.description} completed! Total Time:{elapsed:.1f}s')



X=IndexedBase('X')
Y=IndexedBase('Y')#  X,Y just used to present the type of different input expression.like some multi-expr:X[{},{}]*Y[{},{}] or add-expr:X[{},{}]+Y[{},{}] 
A=IndexedBase('A')# A used to identify the exist of it in expression.
xi=IndexedBase(chr(958))
lamda=IndexedBase(chr(955))
delta=IndexedBase(chr(948))
n=IndexedBase('n')


class SimplifyRule:
    def xiRule(exp):
        expArgs=exp.args#表达式按照加减分成元组不同元素
        res=0
        progress = ProgressBar(len(expArgs), "xiule applying")
        for i in expArgs:
            elemArgs=i.args#表达式按照乘分成元组的不同元素
            subexp=1
            for j in elemArgs:
                if (j!=-1 and j.base==xi):
                    superscript=j.args[1]
                    subscript=j.args[2]
                    j=lamda[superscript,subscript]-delta[superscript,subscript]
                subexp*=j
            res+=subexp
            progress.update()
        
        return simplify(res)

    def natRule(exp):
        expArgs=exp.args#表达式按照加减分成元组不同元素
        res=0 
        progress = ProgressBar(len(expArgs), "natRule applying")
        for i in expArgs:
            elemArgs=i.args#表达式按照乘分成元组的不同元素
            subexp=1
            for j in elemArgs:
                if (j!=-1 and j.base==lamda and len(j.args[1])==1):
                    superscript=j.args[1]
                    subscript=j.args[2]
                    j=n[(),superscript]*delta[superscript,subscript]
                subexp*=j
            res+=subexp
            progress.update()
                
        return simplify(res)


class Filter:
    def filterbody(terms,bodyType):
        if (type(terms)!=type(X[{},{}]+Y[{},{}])):#input terms like A[{},{}]*B[{},{}] or A[{},{}]
            if(type(terms)==type(X[{},{}]*Y[{},{}])):#like A[{},{}]*B[{},{}] or -A[{},{}]*B[{},{}]
                if (type(terms.args[0])==type(A[{},{}])):#if input terms like A[{},{}]*B[{},{}]
                    firstTerm=terms.args[0]
                    if(firstTerm.base==A and len(firstTerm.args[1])==bodyType ):
                        return terms
                    else: return 0
                elif(type(terms.args[1])==type(A[{},{}])):#if input terms like -A[{},{}]*B[{},{}]
                    secondTerm=terms.args[1]
                    if(secondTerm.base==A and len(secondTerm.args[1])==bodyType ):
                        return terms
                    else: return 0    
                elif(bodyType==0):# 0-body term
                    return terms
                else: return 0 
            else:# like  A[{},{}]
                if(terms.base==A and len(terms.args[1]==bodyType)):
                    return terms
                elif(bodyType==0):
                    return terms 
                else: 
                    return 0

        else:#terms like A[{},{}]*B[{},{}] + C[{},{}]*D[{},{}]
            res=0
            progress = ProgressBar(len(terms.args), "filtering")
            for i in terms.args:# i:(A[{},{}]*B[{},{}] , C[{},{}],-D[{},{}]*E[{},{}])
                if(type(i)==type(X[{},{}]*Y[{},{}])):#i like A[{},{}]*B[{},{}] or -A[{},{}]*B[{},{}]
                    if (type(i.args[0])==type(A[{},{}])):#if input terms like A[{},{}]*B[{},{}]
                        firstTerm=i.args[0]
                        if(firstTerm.base==A and len(firstTerm.args[1])==bodyType ):
                            res+=i
                        elif(firstTerm.base!=A and bodyType==0):
                            res+=i
                    elif(type(i.args[1])==type(A[{},{}])):#if input terms like -A[{},{}]*B[{},{}]
                        secondTerm=i.args[1]
                        if(secondTerm.base==A and len(secondTerm.args[1])==bodyType ):
                            res+=i
                        elif(secondTerm.base!=A  and bodyType==0):
                            res+=i
                    elif(bodyType==0):
                        res+=i
                else:# like  A[{},{}]
                    if(i.base==A and len(i.args[1])==bodyType):
                        res+=i
                    elif(i.base!=A and bodyType==0):
                        res+=i
                progress.update()
            return res


def _sepatate(tmpCom,otherTerms):
    '''
    Select terms from otherTerms which has tmpCom term  to removeComTerms.
    Store term to other terms to otherTerms.
    '''
    otherComTerms=0
    if type(otherTerms)==type(X[{},{}]+Y[{},{}]):
        otherTerms=otherTerms.args
    else: otherTerms=set([otherTerms])

    for i  in otherTerms:#(-C*D,E*F)
        for j  in i.args:#(-1,C,D)
            if (j==tmpCom):
                otherComTerms+=i/tmpCom
                break
    return otherComTerms

def uniteSimilarTerms(exp):
    if(type(exp)!=type(X[{},{}]+Y[{},{}])):# return exp when it don't have add terms
        return exp
    baseTerms=exp.args[0]# select A*B from A*B+C*D
    otherTerms=exp-baseTerms
    for tmpCom in baseTerms.args:#turn A*B to (A,B),term=A,B
        com=tmpCom
        comTermsDivideCom=baseTerms/com
        otherComTermsDivideCom=_sepatate(tmpCom,otherTerms)

        if(otherComTermsDivideCom!=0):
            comTermsDivideCom+=otherComTermsDivideCom
            otherTerms=exp-(comTermsDivideCom*com).expand()
            break
    res=com*(uniteSimilarTerms(comTermsDivideCom))+uniteSimilarTerms(otherTerms)#recurse it. 
    return res


def _tupleMultTosimp(tupleUp,tupleLo):
    '''
    Remove the repetitive index of element in Up and Lo.
    '''
    up=[]
    lo=[]
    for i in tupleUp:
        up.append(symbols(str(i)[0]))
        indicesSet.add(str(i)[0])
    for i in tupleLo:
        lo.append(symbols(str(i)[0]))
        indicesSet.add(str(i)[0])
    return tuple(up),tuple(lo)


def indicesMultToSimp(exp):
    ''''
    The indices of canonicalize result are repetitive. This function is designed to simplify the indices.  
    '''
    global indicesSet
    indicesSet=set()
    if (type(exp)!=type(X[{},{}]+Y[{},{}])):#input exp like A[{},{}]*B[{},{}] or A[{},{}]
        res=1
        if(type(exp)==type(X[{},{}]*Y[{},{}])):#like A[{},{}]*B[{},{}] or -A[{},{}]*B[{},{}]
            for i in exp.args:
                if(type(i)==type(A[{},{}])):
                    res*=i.base[_tupleMultTosimp(i.args[1],i.args[2])]
                else: res*=i
        else:# like  A[{},{}] or -A[{},{}] #Impossibly occur because commute Terms would never have such terms
            i=exp
            res=i.base[_tupleMultTosimp(i.args[1],i.args[2])]
    else:#exp like A[{},{}]*B[{},{}] + C[{},{}]*D[{},{}]-E[]F[]
        res=0
        for i in exp.args:#i:(A[{},{}]*B[{},{}] ,   C[{},{}],   -D[{},{}]*E[{},{}])
            if(type(i)==type(X[{},{}]*Y[{},{}])):#i like A[{},{}]*B[{},{}] or -A[{},{}]*B[{},{}]
                tmp=1
                for j in i.args:#j (-1,A,B)
                    if(type(j)==type(A[{},{}])):
                        tmp*=j.base[_tupleMultTosimp(j.args[1],j.args[2])]
                    else:tmp*=j
                res+=tmp
            else:# likeA[{},{}]
                res+=i.base[_tupleMultTosimp(i.args[1],i.args[2])]
    indicesList=list(indicesSet)
    indicesList.sort()
    indicesSymbols=[]
    for i in indicesList:
        indicesSymbols.append(symbols(i))
    
    return res,tuple(indicesSymbols)

def get_all_indices(expr):
    """提取表达式中所有指标"""
    indices_set = set()

    for term in preorder_traversal(expr):
        if isinstance(term, Indexed):
            for i in term.indices[0]: # get up indices
                indices_set.add(i)
            for i in term.indices[1]: # get down indices
                indices_set.add(i)

    return sorted(indices_set, key=str)
            
def get_all_indices(expr):
    """
    get all indices from expression
    """
    indices_set = set()

    for term in preorder_traversal(expr):
        if isinstance(term, Indexed):
            for i in term.indices[0]: # get up indices
                indices_set.add(i)
            for i in term.indices[1]: # get down indices
                indices_set.add(i)

    return sorted(indices_set, key=str)






