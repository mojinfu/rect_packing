import random
import json
def GetTestData(w,h):
    out = []
    for i in range(100):
        ask = []
        for j in range(random.randrange(50,1000)):    
            ask.append([random.randrange(1,w) ,random.randrange(1,h)])
        out.append(ask)
    return out
if __name__ == '__main__': 
    # Step 1: init BrainDQN
    random.seed(0)
    print(json.dumps(GetTestData(20,10)))
    # trainModel()
    