import time

class tic():
    
    def __init__(self):
        self.cur = time.time()
    
    def __call__(self):
        self.cur = time.time()        
        
    def toc(self, setting='secs'):
        if setting == 'secs':
            print("%.2f %s elapsed" % (time.time() - self.cur, setting))
        elif setting == 'mins':
            print("%.2f %s elapsed" % ((time.time() - self.cur) / 60, setting))
        