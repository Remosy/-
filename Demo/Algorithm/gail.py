class GAIL():
    def __init__(self,xprtTraj,initPolicy, policy, w,numIteration)-> None:
        self.xprtTraj = xprtTraj
        self.initPlicy = initPolicy
        self.policy = policy
        self.w = w
        self.numIntaration = numIteration

    def sample(self):

        print("")

    def update(self):
        print("")

    def policyStep(self):
        print("")

    def train(self):
        for x in range(0,self.numIntaration):
            self.sample()
            self.update()
            self.policyStep()


