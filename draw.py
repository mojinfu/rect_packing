
import random
import pygame
import time
class Paper():
    def __init__(self,width,heigth):
        pygame.init()
        self.scal  = int(500 / width) # 扩大倍数
        self.sW = width * self.scal * 2 
        self.sH = (heigth)*self.scal *2
        sW = self.sW
        sH = self.sH
        self._sceen=pygame.display.set_mode([sW ,sH + 500 ])
        self._sceen.fill([255,255,255])
        pygame.draw.lines(self._sceen,[0,0,0],False,[(sW/2 , 0 ),(sW/2 , sH ) ])
        pygame.draw.lines(self._sceen,[0,0,0],False,[(0,sH),(sW,sH) ])
        pygame.draw.lines(self._sceen,[0,0,0],False,[(0,sH/2),(sW,sH/2) ])
        # self._sceen2=pygame.display.set_mode([width * self.scal ,(heigth+1000)*self.scal])
        # self._sceen2.fill([255,255,255])
        # pygame.draw.lines(self._sceen,[0,0,0],False,[(0* self.scal,heigth* self.scal),(width * self.scal ,heigth * self.scal) ])


    def AddRect(self, x,y,width,heigth,binChose):

        X = x* self.scal
        Y = y* self.scal
        if binChose == 1 or binChose ==3:
            X = X + self.sW/2
        if binChose == 2 or binChose ==3:
            Y = Y + self.sH/2
        pygame.draw.lines(self._sceen,[255,0,0],True,[(X,Y),(X + width* self.scal,Y) , (X + width* self.scal,Y + heigth* self.scal),(X ,Y + heigth* self.scal)])
        # pygame.draw.rect(self._sceen,[random.randint(0,255),random.randint(0,255),random.randint(0,255)],[x* self.scal,y* self.scal,width* self.scal,heigth* self.scal])
        pygame.display.flip()
    def AddBlackRect(self, x,y,width,heigth,binChose):
        X = x* self.scal
        Y = y* self.scal
        if binChose == 1 or binChose ==3:
            X = X + self.sW/2
        if binChose == 2 or binChose ==3:
            Y = Y + self.sH/2
        pygame.draw.lines(self._sceen,[0,0,0],True,[(X,Y),(X + width* self.scal,Y) , (X + width* self.scal,Y + heigth* self.scal),(X ,Y + heigth* self.scal)])
        pygame.display.flip()
    def Reset(self):
        self._sceen.fill([255,255,255])
        pygame.display.flip()
    def Close(self):
        pygame.quit()
if __name__ == '__main__':
    paper = Paper(640,480)
    paper.AddRect(0,0,50,100)
    time.sleep(5)
    paper.Close()