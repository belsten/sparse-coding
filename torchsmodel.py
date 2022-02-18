import numpy as np
import torch 
from torch.utils.data import DataLoader
import pickle as pkl

class sparsecoding(torch.nn.Module):
    
    def __init__(self,n_basis,n,lmbda=0.2,eta=1e-2,device=None,**kwargs):
        """
        Parameters:
        n_basis - scalar (1,)
            size of dictionary
        n - scalar (1,)
            number of features
        lmbda - scalar (1,) default=0.2
            update rate - NOTE: unused with LCA algo. See thresh
        eta - scalar (1,) default=1e-2
            step size of dynamical eqn
        thresh - scalar (1,) default=1e-2
            threshold for shrink prior
        stop_early - boolean defualt=False
            evaluate stopping criteria using eps
        eps - scalar (1,) default=1e-2
            stopping criteria of norm of difference 
            between coefficient update - eps or n_iter, whichever 
            comes first
        n_itr - scalar (1,) default=100
            number of iterations to update coefficient estimate
        nabla - scalar (1,) default=1e-2
            learning rate of dictionary update
        device - torch.device defualt=torch.device("cpu")
            which device to utilize
        dict_decay - scalar (1,) default=None
            scalar coefficient of L2 weight decay on dictionary.
            If equal to None, no weight decay applied
        """
        super(sparsecoding, self).__init__()
        self.n_basis    = n_basis
        self.n          = n
        self.device     = torch.device("cpu") if device is None else device
        self.n_itr      = kwargs.pop('n_itr',100)
        self.eps        = kwargs.pop('eps', 1e-3)
        self.stop_early = kwargs.pop('stop_early', False)   
        self.dict_decay = kwargs.pop('dict_decay', None)   
        self.thresh     = torch.tensor(np.float32(kwargs.pop('thresh',1e-2))).to(self.device) 
        self.nabla      = torch.tensor(np.float32(kwargs.pop('nabla',1e-2))).to(self.device)            
        self.eta        = torch.tensor(np.float32(eta)).to(self.device)
        self.lmbda      = torch.tensor(np.float32(lmbda)).to(self.device)
        self.D          = torch.randn((self.n, self.n_basis)).to(self.device)
        self.normalizedict()

    
    def ista(self,I):
        """
        Infer coefficients for each image in I made up of dict elements D
        Method implemented according to 1996 Olshausen and Field
        ---
        Parameters:
        I - torch.tensor (batch_size,n)
            input images
        ---
        Returns:
        A - scalar (batch_size,n_basis)
            sparse coefficients
        """
        batch_size = I.size(0)

        # initialize
        A = torch.zeros((batch_size,self.n_basis)).to(self.device)
        residual = I - torch.mm(self.D,A.t()).t()
       
        for i in range(self.n_itr):
            
            A = A.add(self.eta * self.D.t().mm(residual.t()).t())
            A = A.sub(self.eta * self.lmbda).clamp(min=0.0)
            if self.stop_early:
                residual_new = I - torch.mm(self.D,A.t()).t()
                if (residual_new - residual).norm(p=2).sum() < self.eps:
                    break
                residual = residual_new
            else:
                residual = I - torch.mm(self.D,A.t()).t()    
            # check for nans
            self.checknan(A,'coefficients')
        return A
    
    
    def lca(self,I):
        """
        Infer coefficients for each image in I made up of dict elements D
        Method implemented according Locally competative algorithm (Rozell 2008)
        ---
        Parameters:
        I - torch.tensor (batch_size,n)
            input images
        ---
        Returns:
        A - scalar (batch_size,n_basis)
            sparse coefficients
        """
        batch_size = I.size(0)

        # initialize
        u = torch.zeros((batch_size,self.n_basis)).to(self.device)
        a = torch.zeros((batch_size,self.n_basis)).to(self.device)

        b = (self.D.t()@I.t()).t()
        G = self.D.t()@self.D-torch.eye(self.n_basis).to(self.device)
        for i in range(self.n_itr):
            if self.stop_early:
                old_u = u.clone().detach()
                
            a = self.Tsoft(u)
            du = b-u-(G@a.t()).t()
            u = u + self.eta*du
            
            if self.stop_early:
                if (old_u - u).norm(p=2).sum() < self.eps:
                    break 
            self.checknan(u,'coefficients')
            
        return self.Tsoft(u)
                
                
    def Tsoft(self,u):
        """
        Soft threshhold function according to Rozell 2008
        
        inputs:
        u - torch tensor (batch_size,n_basis)
            membrane potentials
        ---
        returns: 
        a - torch tensor (batch_size,n_basis)
            activations
        """
        a = (torch.abs(u) - self.thresh).clamp(min=0.)
        a = torch.sign(u)*a
        return a
        
                

    def updatedict(self,I,A):
        """
        Compute gradient of energy function w.r.t. dict elements, and update 
        ---
        Parameters:
        I - scalar (batch_size,n)
            input images, n_images usually batch size
        A - scalar (batch_size,n_basis)
            alread-inferred coefficients
        """
        residual = I - torch.mm(self.D,A.t()).t()
        dD = torch.mm(residual.t(),A)
        if self.dict_decay != None:
            dD = dD + self.dict_decay*self.D
        self.D = torch.add(self.D, self.nabla*dD)
        self.checknan()
        
        
    def normalizedict(self):
        """
        Normalize columns of dictionary matrix D s.t. 
        """
        self.D = self.D.div_(self.D.norm(p=2,dim=0))
        self.checknan()
        
        
    def learndict(self,dataset,n_epoch,batch_size):
        """
        Learn dictionary for nepoch
        ---
        Parameters:
        dataset - torch.utils.data.Dataset 
            input dataset class
        n_epoch - scalar (1,)
            number of iterations to learn dictionary
        batch_size - scalar (1,)
            batch size to learn on
        ---
        Returns:
        loss - scalar (nepoch)
        """
        loss = []
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        iterloader = iter(dataloader)
        for i in range(n_epoch):
            try:
                batch = next(iterloader)
            except StopIteration:
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                iterloader = iter(dataloader)
                batch = next(iterloader)
            # infer coefficients
            A = self.lca(batch)
            # update dictionary
            self.updatedict(batch,A)
            # normalize dictionary
            self.normalizedict()
            l = (batch-torch.mm(self.D,A.t()).t()).norm(dim=1).square().sum()+(self.lmbda*A).sum()
            loss.append(l.to(self.device).cpu().detach().numpy())
        return np.asarray(loss)
        
        
    def getnumpydict(self):
        """
        return dictionary as numpy array
        """
        return self.D.cpu().detach().numpy()
    
    
    def checknan(self,data=torch.tensor(0),name='data'):
        """
        Check for nan values in dictinary, or data
        ---
        Parameters:
        data - torch.tensor default=1
            data to check for nans
        name - string
            name to add to error, if one is thrown
        """
        if torch.isnan(data).any():
            raise ValueError('sparsecoding error: nan in %s.'%(name))
        if torch.isnan(self.D).any():
            raise ValueError('sparsecoding error: nan in dictionary.')
            
            
    def loaddict(self,filename):
        '''
        Load dictionary from pkl dump
        ---
        filename - string
            file to load as self.D
        '''
        file = open(filename,'rb')
        nD = pkl.load(file)
        file.close()
        self.D = torch.tensor(nD.astype(np.float32)).to(self.device) 
        
            
    def savedict(self,filename):
        '''
        Save dictionary to pkl dump
        ---
        filename - string
            file to save self.D to
        '''
        filehandler = open(filename,"wb")
        pkl.dump(self.getnumpydict(),filehandler)
        filehandler.close()

        