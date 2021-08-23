class Postion:
    def __init__(self,x,y):
        self._x=x
        self._y=y
class Item:
    def __init__(self,width,height,index,name):
        self.width=width
        self.height=height
        self._index=index
        self._name=name
class Rectangle:
    def __init__(self,postion,item):
        self._postion=postion
        self._item=item
        self._ifRotated = False
    def Rotate(self) :
        if self._ifRotated :
            self._ifRotated = False
        else :
            self._ifRotated = True
    def Width(self) :    
        if self._ifRotated:
            return self._item.height
        else:
            return self._item.width
    def Height(self) :    
        if self._ifRotated:
            return self._item.width
        else:
            return self._item.height
    def X(self) :   
        return self._postion._x
    def Y(self) :    
        return self._postion._y
    def IfRotate(self) :
	    return self._ifRotated
    def Area(self) :
	    return self._item.width * self._item.height


if __name__ == '__main__':
    p=Postion(0,0)
    it=Item(1,5,0,"it")
    rect=Rectangle(p,it)
    print(rect.Area())