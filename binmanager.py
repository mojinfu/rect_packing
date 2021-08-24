from rect import *
from draw import *
import random
import time
import numpy
class env:
    def __init__(self,binWidth,binHeight,ifRotationAllowed=False,binManagerName=''):
        self.binWidth=binWidth
        self.binHeight=binHeight
        self.ifRotationAllowed=ifRotationAllowed
        self.binManagerName=binManagerName
class freeRects:
    def __init__(self,x,y,width,height):
        self.X      = x      
        self.Y      = y      
        self.width  = width  
        self.height = height 
    def Area(self):
        return self.X*self.Y
    def CanPlace(self ,w,h):
        if self.width < w or self.height<h:
            return False
        return True
    def CutOverLap(self ,pInfo ):#todo
        x1,x2,y1,y2 = pInfo[0],pInfo[1],pInfo[2],pInfo[3]
        lt = []
        lt.append( freeRects(self.X,self.Y,self.width , y1 - self.Y) ) 
        lt.append( freeRects(self.X,y2,self.width , self.Y+self.height - y2) )
        lt.append( freeRects(self.X,self.Y,x1 - self.X ,self.height) )
        lt.append( freeRects(x2 ,self.Y,self.X +self.width - x2  ,self.height) )
        finallt = []
        for fr in (lt):
            if fr.width < 0 or fr.height < 0:
                raise("dlkldk")
            if fr.width != 0 and fr.height!=0:
                finallt.append(fr)
                rect = Rectangle(Postion(x1,y1),Item(x2-x1,y2-y1 , 1,"sd") )
                isO , temp = fr.OverLap(rect)
                if isO:
                    print(rect.X() , rect.Y() , rect.Width(), rect.Height())
                    raise(" ks")
        return finallt
    def OverLap(self,rect):
        Ax1 = self.X
        Ax2 = self.X+self.width

        Ay1 = self.Y
        Ay2 = self.Y+self.height


        Bx1 = rect.X()
        Bx2 = rect.X()+rect.Width()

        By1 = rect.Y()
        By2 = rect.Y()+rect.Height()



        if Ax2 <= Bx1 or Ax1>=Bx2:
            return False ,[]

        if Ay2 <= By1 or Ay1 >= By2:
            return False ,[]
        Ox1 = max(Ax1 , Bx1)
        Ox2 = min(Ax2 , Bx2)
        Oy1 = max(Ay1 , By1)
        Oy2 = min(Ay2 , By2)
        # if Ox2 >= Ax2 or Oy2 >= Ax2 :
        #     raise("dkl")

        if not(Oy1 >=Ay1 and  Oy1 >=By1 and Oy2  <=Ay2 and  Oy2 <= By2):
            raise("dklss")
        if not(Ox1 >=Ax1 and  Ox1 >=Bx1 and Ox2  <=Ax2 and  Ox2 <= Bx2):
            raise("dklss")
        return True, [Ox1 ,Ox2 ,Oy1 ,Oy2]

    def Place(self ,  xx,w,h):
        if self.width < w or self.height<h:
            raise("")
        lt = []
        placeX = -1
        placeY = -1
        if xx == 1:
            lt= [freeRects(self.X + 0,self.Y+0,self.width-w,self.height) , freeRects(self.X + 0,self.Y+h, self.width, self.height- h )]
            placeX = self.width-w
            placeY = 0
        if xx == 2:
            lt= [freeRects(self.X +  0,self.Y+0,self.width ,self.height - h),freeRects(self.X + 0,self.Y+0, self.width - w , self.height )]
            placeX = self.width-w
            placeY = self.height-h
        if xx == 3:
            lt= [freeRects(self.X +  0,self.Y+0,self.width ,self.height - h),freeRects(self.X + w, self.Y, self.width - w , self.height)]
            placeX = 0
            placeY = self.height-h
        if xx == 4:
            lt= [freeRects(self.X +  0, self.Y+ h ,self.width , self.height- h ),freeRects(self.X + w ,self.Y+ 0, self.width-w, self.height )]
            placeX = 0
            placeY = 0
        finallt = [] 
        for fr in (lt):
            if fr.width != 0 and fr.height!=0:
                finallt.append(fr)
        return finallt , placeX+self.X, placeY+self.Y

class bin:
    def __init__(self,width, height ,name=""):
        self.width      = width
        self.height =   height 
        self.name   =name
        self.UnUsedArea   = width * height
        self.placedRect =  []
        self.freeRects = [freeRects(0,0,width,height)]
    def ChooseAlgo_MinArea(self,item):
        minArea = 0 
        frI = -1
        for i,fr in enumerate(self.freeRects):
            if fr.CanPlace(item.width,item.height):
                if minArea ==0 :
                    minArea = fr.Area()
                    frI = i
                if minArea > fr.Area():
                    minArea = fr.Area()
                    frI = i
        return frI
    def ChooseAlgo_MinXMinY(self,item):#MinX MinY
        minX = -1 
        minY = -1
        frI = -1
        for i,fr in enumerate(self.freeRects):
            if fr.CanPlace(item.width,item.height):
                if minX <0 :
                    minX = fr.X
                    minY = fr.Y
                    frI = i
                if minX > fr.X:
                    minX = fr.X
                    minY = fr.Y
                    frI = i
                elif minX == fr.X and minY > fr.Y:
                    minX = fr.X
                    minY = fr.Y
                    frI = i
        return frI

    def ChooseAlgo_MinYMinX(self,item):#MinX MinY
        minX = -1 
        minY = -1
        frI = -1
        for i,fr in enumerate(self.freeRects):
            if fr.CanPlace(item.width,item.height):
                if minY <0 :
                    minY = fr.Y
                    minX = fr.X
                    frI = i
                if minY > fr.Y:
                    minY = fr.Y
                    minX = fr.X
                    frI = i
                elif minY == fr.Y and minX > fr.X:
                    minY = fr.Y
                    minX = fr.X
                    frI = i
        return frI


    def place(self ,  item ):
        frI = self.ChooseAlgo_MinXMinY(item)
        if frI>=0:
            lt,x,y = self.freeRects[frI].Place( 4,item.width,item.height)
            # lt,x,y = self.freeRects[frI].Place( random.randint(1,4),item.width,item.height)
            # for placeCheck in self.placedRect:
            #     for leftFr in lt:
            #         check,temp =   leftFr.OverLap(placeCheck )
            #         if check:
            #             raise(" 3 ")   
            del self.freeRects[frI]
            rect = Rectangle(Postion(x,y),item )
            for fr in self.freeRects:
                # for placeCheck in self.placedRect:
                #     check,temp =   fr.OverLap(placeCheck )
                #     if check:
                #         raise(" 1 ")   

                ifOver, over= fr.OverLap(rect)
                if ifOver:
                    # print("over:",over)
                    # self.freeRects.remove(fr)
                    newFree = fr.CutOverLap(over)
                    # for cutBornFree in newFree:
                    #     # print("newFree :",cutBornFree.X , cutBornFree.Y , cutBornFree.width ,cutBornFree.height)
                    #     for placeCheck in self.placedRect:

                    #        check,temp =   cutBornFree.OverLap(placeCheck )
                    #        if check:
                    #            raise(" ")
                    lt.extend(newFree)
                else:
                    lt.append(fr)
                
            self.freeRects = lt
            # for placeCheck in self.placedRect:
            #     for leftFr in self.freeRects:
            #         check,temp =   leftFr.OverLap(placeCheck )
            #         if check:
            #             raise(" 2 ")   
            self.placedRect.append(rect)
            self.UnUsedArea = self.UnUsedArea - item.height*item.width
            return True
        else: 
            return False
def min(a,b):
    if a<b:
        return a
    return b
def max(a,b):
    if a<b:
        return b
    return a
class binManager:
    def __init__(self,env,algoVBinLimit=5,algoVItemLimit = 10):
        self.env  = env
        self.forget = False
        self.items= []
        self.placed= []
        self.bins = [bin(env.binWidth ,env.binHeight)]#已经创建的bin
        self.algoVBinLimit  = algoVBinLimit
        self.algoVItemLimit  = algoVItemLimit
        self._algoVBinIndexList= [0]# 长度可以作为算法输入 算法视野bins
        self._algoVItemIndexList= []# 长度可以作为算法输入 算法视野items
        self._maxIndexInunplacedItemIndexList = 0# 长度可以作为算法输入 往后遍历找出视野外的items
        # self._algoVBinLimit = 10
        for i in range(self.algoVBinLimit-1):
            self._algoVBinIndexList.append(-1)

        for i in range(self.algoVItemLimit):
            if i<= len(self.items) -1:
                self._algoVItemIndexList.append(i)
                self._maxIndexInunplacedItemIndexList = i
            else:
                self._algoVItemIndexList.append(-1)
        self.placedNum = 0
       
        m = numpy.zeros((env.binHeight,env.binWidth))
        for i in range(env.binWidth):
            for j in range(env.binHeight):
                            m[j,i] = -1
        self._negaM = m
    def AddRandomItem(self, name="" ):
        self.AddItem( random.randint(1, self.env.binWidth)  ,random.randint(1, self.env.binHeight) )
        # self.AddItem( random.randint(1, self.env.binWidth)  ,random(1, self.env.binHeight) )
    def AddItem(self,width, height , name="" ):
        it = Item(width ,height, len(self.items),name)
        self.items.append(it)
        rePlace = -1
        for i in  range(len(self._algoVItemIndexList)):
            if self._algoVItemIndexList[i] <0:
                
                self._algoVItemIndexList[i] = len(self.items)-1
                if self._maxIndexInunplacedItemIndexList < self._algoVItemIndexList[i]:
                    self._maxIndexInunplacedItemIndexList   =    self._algoVItemIndexList[i]
                break
            
        self.placed.append(0)
    def Action(self, itAlgoVIndex, binAlgoVIndex,rotationChose):
        if itAlgoVIndex < 0  or itAlgoVIndex >= len(self._algoVItemIndexList)  or binAlgoVIndex < 0   or binAlgoVIndex >= len(self._algoVBinIndexList) :
            return False,False, -1
        itIndex = self._algoVItemIndexList[itAlgoVIndex]
        binIndex = self._algoVBinIndexList[binAlgoVIndex]
        if binIndex<0:
            self.bins.append(bin(self.env.binWidth ,self.env.binHeight))
            self._algoVBinIndexList[binAlgoVIndex] = len(self.bins) -1 
            binIndex = self._algoVBinIndexList[binAlgoVIndex]
           
        if itIndex<0:
            raise("")
            # return False,False, -1  
        oldItemIndex = itIndex    
        ifComplete,ifSuccess=  self.Place(itIndex,binIndex,rotationChose)
        self._algoVItemIndexList[itAlgoVIndex] = -1
        if self.placedNum % len(self._algoVItemIndexList) ==0:
            for iii  in range(len(self._algoVItemIndexList)):
                if  self._algoVItemIndexList[iii] == -1:
                    if self._maxIndexInunplacedItemIndexList < len(self.items) - 1:
                        self._maxIndexInunplacedItemIndexList = self._maxIndexInunplacedItemIndexList +1
                        self._algoVItemIndexList[iii] = self._maxIndexInunplacedItemIndexList
        if not ifSuccess:
            oldBinIndex = self._algoVBinIndexList[binAlgoVIndex]
            replacebin = binAlgoVIndex
            for i in range( len(self._algoVBinIndexList)):
                if self._algoVBinIndexList[i] < 0 :
                    replacebin = i 
                    break
            self._algoVBinIndexList[replacebin] = len(self.bins) -1 
            # return ifComplete,ifSuccess,-1 * self.bins[oldBinIndex].UnUsedArea
            return True,ifSuccess,-1000
        else:
            # return ifComplete,ifSuccess, 0
            return ifComplete,ifSuccess, self.items[oldItemIndex].width * self.items[oldItemIndex].height

    def Place(self, itIndex, binIndex,rotationChose):
        # print("Put ",itIndex ,"in ",binIndex)
        ifComplete = False 
        ifSuccess = False    
        if self.placedNum == len(self.items)-1:
            ifComplete =  True
        self.placedNum = self.placedNum + 1 
        # print("placedNum:",self.placedNum)
        self.placed[itIndex] = 1
        itemAfterRotation = self.items[itIndex]
        if rotationChose==1:
            itemAfterRotation = Item(self.items[itIndex].height,self.items[itIndex].width,self.items[itIndex]._index,self.items[itIndex]._name )

        if self.bins[binIndex].place(itemAfterRotation):
            # _algoVItemIndexList
            # _maxIndexInunplacedItemIndexList
            ifSuccess = True
            return ifComplete,ifSuccess 
        else :
            ifSuccess = False
            # _algoVBinIndexList
            # _algoVItemIndexList
            # _maxIndexInunplacedItemIndexList
            self.bins.append(bin(self.env.binWidth ,self.env.binHeight))
            self.bins[-1].place(itemAfterRotation)
            return ifComplete,ifSuccess

    def AllBinStausPics(self):
        for bin in self.bins:
            paper = Paper(bin.width,bin.height)
            for rect in bin.placedRect:
                # print(rect.X(),rect.Y(),rect.Width(),rect.Height())
                paper.AddRect(rect.X(),rect.Y(),rect.Width(),rect.Height())
                time.sleep(0.5)
            paper.Close()
        return

    def BinStausMixs(self):
        mList = []
        for binI in self._algoVBinIndexList:
            if binI>= 0:
                bin = self.bins[binI]
                m = numpy.zeros((bin.height,bin.width))
                for rect in bin.placedRect:
                    x,y= rect.X(),rect.Y()
                    for i in range(rect.Width()):
                        for j in range(rect.Height()):
                            m[j+y,i+x] = 1
                mList.append(m)
            else:
                mList.append(self._negaM)
        return mList


    def ItemMixs(self):
        out = []
        for itI  in self._algoVItemIndexList:
            if itI<0:
                out.append([-1,-1])
            else: 
                out.append([self.items[itI].width,self.items[itI].height])
        return out
    def AllStatus(self):
        out = []
        out.append(self.BinStausMixs())
        out.append(self.ItemMixs())
        return out
if __name__ == "__main__":
    random.seed(0)
    env =env(20,10)
    bm= binManager(env)
    for i in range(1000):
        bm.AddRandomItem()
    while True:
        if bm.placedNum == len(bm.items) -3:
            break
        else:
            bm.Action(random.randint(0,bm.algoVItemLimit - 1),random.randint(0,bm.algoVBinLimit - 1))
    for i in (bm.AllStatus()):
        print(i)
    bm.AllBinStausPics()