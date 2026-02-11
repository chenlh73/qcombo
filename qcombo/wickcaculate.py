'''
This module includes operations for handling Wick's theorem cases.
'''
import sympy as sy
from sympy.utilities.iterables import multiset_partitions  # For set partitions
from itertools import permutations  # For permutations
from sympy import IndexedBase  # For storing indices
from .tools import ProgressBar


# Alpha=IndexedBase(chr(913))
Alpha=IndexedBase('A')
xi=IndexedBase(chr(958))
lamda=IndexedBase(chr(955))
delta=IndexedBase(chr(948))
n=IndexedBase('n')

class _ListOperations:
    '''
    Some important operations to process the list. 
    
    '''

    def transpose(listObject)->list:
        '''
        Transpose a list.

        '''
        #Use numpy to trans it may be more convenient.But in This way, we dont need extra package.
        res=list(map(list,zip(*listObject)))
        return  res
    
    def signature(listObject)->int:
        '''
        Make sure that the listObjetct is a 1 dimension list.
        If the list is odd permutation,return -1;Otherwise ,return 1.
        '''
        sig=1
        for i in range(len(listObject)):
            for listObject_j in listObject[:i]:
                if(str(listObject_j)>str(listObject[i])):
                    sig=(-1)*sig
        return sig
    
    def sort(listObject)->list:
        '''
        Sort the list by the length of it element. If the element is list, sort the listObject in the length of it's sublist. 
        '''
        listObject.sort(key=len,reverse=False)
        #lambda listObject:listObject.sort(key=len,reverse=False)
        return listObject

    def union(listObject)->list:
        '''
        remove the same elements of listObject and keep one of them in the listObject.
        '''
        res=[]
        for listObject_i in listObject:
            if listObject_i not in res:
                res.append(listObject_i)
        return res

    def setPartitions(listObject)->list:
        '''
        Return unique partitions of the given multiset (in list form)

        # Examples
        >>> setPartitions([1, 1, 2])
        [[[1, 1, 2]], [[1, 1], [2]], [[1, 2], [1]], [[1], [1], [2]]]

        # Note
        Used `multiset_partitions` in  `sympy.utilities.iterables` to achieve setPartitions. 

        '''

        res=[]
        for i in multiset_partitions(listObject):# trans multiset_partitions object to list
            res.append(i)
        return res

    def sortPartitions(listObject)->list:
        '''
        Sort the list by the length of it's elements.Meanwhile sort it elemnts.

        # Examples
        >>> sortPartitions([[[1,2],[1,2,3],[1]],[[1,2,3,4],[1,2,3]]])
        [[[1, 2, 3], [1, 2, 3, 4]], [[1], [1, 2], [1, 2, 3]]]
        >>> sortPartitions([[[1,2],[1,2,3],[1]]])
        [[[1], [1, 2], [1, 2, 3]]]    

        '''
        res=listObject
        if(len(res)>=1):
            res.sort(key=len,reverse=False)
            for i in res:
                i.sort(key=len,reverse=False)
        return res 
 

class Wick:
    '''
    
    '''

    def __init__(self,LUpLo,RUpLo):
        self._LUp,self._LLo,self._RUp,self._RLo=LUpLo[0],LUpLo[1],RUpLo[0],RUpLo[1]
        self.gw=0#generalizedWick result
        self.cmt=0#commutate result
    
    def _MatchPartitions(self,pUp,pLo):
        '''
        将划分出的上标和下标的长度一样进行匹配。并且要求内部的子表的长度也要一样，这样才能形成符号的匹配。
        '''
        res=[]
        for pUp_i in pUp:
            lengthPartitionUp=len(pUp_i)
            for pLo_j in pLo:
                lengthPartitionLo=len(pLo_j)
                if(lengthPartitionUp==lengthPartitionLo):
                    strcutPartitionUp=list(map(len,pUp_i))#返回列表中子表长度
                    strcutPartitionLo=list(map(len,pLo_j))
                    if(strcutPartitionUp==strcutPartitionLo):#要求列表中的子表的长度也要相同才行。
                        res.append(_ListOperations.transpose([pUp_i,pLo_j]))   
        return res


    def _EvaluateConstraction(self,full_i_k,LUp,LLo,RUp,RLo):
        '''
        evaluate a contraction for a specific combination of indices
        '''
        if(((set(full_i_k[0]) & set(LUp)==set(full_i_k[0])) and (set(full_i_k[1]) & set(LLo)==set(full_i_k[1])))  or  ((set(full_i_k[0]) & set(RUp)==set(full_i_k[0])) and (set(full_i_k[1]) & set(RLo)==set(full_i_k[1])))):
            res=0
        else:
            if(len(full_i_k[0])==1 and len(full_i_k[1])==1  and  (set(full_i_k[0]) & set(RUp)==set(full_i_k[0])) and (set(full_i_k[1]) & set(LLo)==set(full_i_k[1]))):
                #res=symbols_laxexp('xi',full_i_k))
                res=xi[tuple(full_i_k[0]),tuple(full_i_k[1])]
            else:
                #res=symbols_laxexp('lambda',full_i_k) 
                res=lamda[tuple(full_i_k[0]),tuple(full_i_k[1])]
        return res


    def _ConsturctTerms(self,part_i,idxUp,idxLo,LUp,LLo,RUp,RLo):#part_i表示此时只是挑选出上下标组合情况的一种，也就是一种输出方式。
        '''
        '''
        sigUp=_ListOperations.signature(idxUp)#确定原来符号的排序情况，应为并不会所有的输入符号会按照递增顺序输入。
        sigLo=_ListOperations.signature(idxLo)
        #转置之后取下标，并排列下标的所有可能排序情况
        #这里对下标进行全排列的原因是：在Step构建出来的组合情况还不是完整的，因为上下标被单增排序过。但实际组合情况可能会有不按照单增顺序来的。
        permutations_part_i=list(map(list,list(permutations(_ListOperations.transpose(part_i)[1]))))#借助了标准包itertools.permutations的全排列函数
        #为什么又要重新排序去掉相同的呢？这样只是具有相同长度的子集在全排列了。
        permutations_part_i=_ListOperations.union(map(_ListOperations.sort,permutations_part_i))#排序并去重
        full=list(map(#这个list是为了将map数据转换为list数据
            _ListOperations.transpose,list(#再次转置，变成每组上下标一个子list#这个list似乎不需要
            map(#对每一个下标可能性per都与上标结合
            lambda permu_i:[_ListOperations.transpose(part_i)[0],permu_i],#将part_i转置得到上标，再与下标结合
            permutations_part_i)
            )
        ))
        expr=0
        for i in range(len(full)):
            termIndexUp= [y for x in [y[0] for y in full[i]] for y in x]#第i个组合的上标有序集合:先取full[i]的所有上标然后再将上标压平成一个数组
            #termIndexUp=list(np.array(full[i],dtype=object)[:,0].flatten())#如果full是numpy数组可以用此方法，但是为了减少对其他包的依赖未采用#这行代码结果是错误的，因为flatten不能对子表长度不同数组进行展开
            termIndexLo= [y for x in [y[1] for y in full[i]] for y in x]
            tmplambda=1
            for j in range(len(full[i])):#len(full[i])是符号个数？
                tmpA=1
                for k in range(len(full[i])):
                    if(j==k):
                        #tmpA*=symbols_laxexp('A',full[i][k])#将指标对应符号A
                        tmpA*=Alpha[tuple(full[i][k][0]),tuple(full[i][k][1])]
                    else:
                        tmpA*=self._EvaluateConstraction(full[i][k],LUp,LLo,RUp,RLo)#对应其他符号xi和lambda
                expr+=_ListOperations.signature(termIndexUp)*_ListOperations.signature(termIndexLo)*tmpA
                tmplambda*=self._EvaluateConstraction(full[i][j],LUp,LLo,RUp,RLo)
            expr+=_ListOperations.signature(termIndexUp)*_ListOperations.signature(termIndexLo)*tmplambda

        expr*=sigUp*sigLo
        return expr  

    
    def generalizedWick(self):
        idxUp=self._LUp + self._RUp
        idxLo=self._LLo + self._RLo
        partUp=_ListOperations.sortPartitions(_ListOperations.setPartitions(idxUp))
        partLo=_ListOperations.sortPartitions(_ListOperations.setPartitions(idxLo))
        part=self._MatchPartitions(partUp,partLo)#part是挑选出来的所有可能的上标和下标的集合划分子集的长度也相互对应情况。这是可能输出的符号的上下标。
        progress = ProgressBar(len(part), "generalize wick caculating")
        for i in range(len(part)):
            self.gw+=self._ConsturctTerms(part[i],idxUp,idxLo,self._LUp,self._LLo,self._RUp,self._RLo)
            progress.update()
    
    def commmutate(self):
        '''
        计算对易式
        '''
        LUp,LLo,RUp,RLo=self._LUp,self._LLo,self._RUp,self._RLo

        positiveWick=Wick([LUp,LLo],[RUp,RLo])
        positiveWick.generalizedWick()
        
        negativeWick=Wick([RUp,RLo],[LUp,LLo])
        negativeWick.generalizedWick()
        self.gw=positiveWick.gw
        self.cmt=positiveWick.gw-negativeWick.gw




if __name__=='__main__':

    a=IndexedBase('a')
    b=IndexedBase('b')
    p=IndexedBase('p')    
    q=IndexedBase('q')
    s=IndexedBase('s')
    r=IndexedBase('r')
    from sympy import false, true

    my_wick_test=Wick([(a,),(b,)],[(p,q),(r,s)])
    my_wick_test.commmutate()

    res=my_wick_test.cmt
    print(res)
