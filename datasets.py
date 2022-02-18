import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


## ================================================================================
#  NATURAL IMAGES
## ================================================================================
class naturalscenes(Dataset):
    
    def __init__(self, img_dir, patch_size, patch_overlap, data_key='IMAGESr',device=None):
        """
        Parameters:
        img_dir - string
            *.mat file to load images from, expected to be of for (pix,pix,n_img)
        patch_size - scalar (1,)
            patch row/column size to extract
        patch_overlap - scalar (1,) default=0
            amount to overlap patches by in pixels. If 0, no overlap
        data_key - string default='IMAGESr'
            key to query mat dict with to get data
        device - torch.device default == cpu
            device to load data on
        ---
        """
        self.device = torch.device("cpu") if device is None else device
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        
        # load image dataset
        images_dict = sio.loadmat(img_dir)
        self.images = np.asarray(images_dict[data_key]) # scalar - (m,m,n_img)
        
        # make patches
        self.extractpatches()
    
    
    def __len__(self):
        '''
        return number of patches
        '''
        return self.patches.shape[0]

    
    def __getitem__(self, idx):
        '''
        return patch at idx
        '''
        return self.patches[idx,:]
    
    
    def extractpatches(self):
        """
        Extracts image patches from images
        ---
        Defines:
        self.patches - scalar torch.tensor (n_patch,patch_dim**2)
            image patches

        Note: if patch_dim doesn't go into pix evenly, 
              no error will be thrown left, bottom part of image cut-off
        """
        pix,_,n_img = self.images.shape

        n = (pix-self.patch_size)//(self.patch_size-self.patch_overlap)

        self.patches = []
        for img in range(n_img):
            for i in range(n):
                for j in range(n):
                    rl = i*(self.patch_size - self.patch_overlap)
                    rr = rl+self.patch_size
                    cl = j*(self.patch_size - self.patch_overlap)
                    cr = cl+self.patch_size
                    self.patches.append(self.images[rl:rr,cl:cr,img])
        self.patches = np.asarray(self.patches,dtype=np.float32).reshape(-1,self.patch_size**2)
        self.patches = torch.from_numpy(self.patches).to(self.device)

        
## ================================================================================
#  COLOR NATURAL IMAGES
## ================================================================================
class colornaturalscenes(Dataset):
    
    def __init__(self,
                 img_dir,
                 patch_size,
                 patch_overlap,
                 data_key='lms',
                 device=None,
                 std_thresh=None,
                 whiten=False):
        """
        McGill dataset of LMS valued color images
        ---
        Parameters:
        img_dir - string
            *.mat files to load images from
        patch_size - scalar (1,)
            patch row/column size to extract
        patch_overlap - scalar (1,) default=0
            amount to overlap patches by in pixels. If 0, no overlap
        data_key - string default='lms'
            key to query mat dictionary with to get data
        device - torch.device default == cpu
            device to load data on
        ---
        """
        import glob
        
        self.device = torch.device("cpu") if device is None else device
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.std_thresh = std_thresh
        self.whiten = whiten
        
        # load image dataset
        matfiles = []
        self.images = []
        for file in glob.glob(img_dir+"/*.mat"):
            matfiles.append(file)
            self.images.append(sio.loadmat(file)[data_key])
        
        # make patches
        self.extractpatches()
    
    
    def __len__(self):
        '''
        return number of patches
        '''
        return self.patches.shape[0]

    
    
    def __getitem__(self, idx):
        '''
        return patch at idx
        '''
        return self.patches[idx,:]
    
    
    
    def extractpatches(self):
        """
        Extracts image patches from images
        ---
        Defines:
        self.patches - scalar torch.tensor (n_patch,patch_dim**2)
            image patches

        Note: if patch_dim doesn't go into pix evenly, 
              no error will be thrown left, bottom part of image cut-off
        """

        self.patches = []
        for img in self.images:
            # img = img[::2,::2]
            n = (img.shape[0]-self.patch_size)//(self.patch_size-self.patch_overlap)
            m = (img.shape[1]-self.patch_size)//(self.patch_size-self.patch_overlap)
            for i in range(n):
                for j in range(m):
                    rl = i*(self.patch_size - self.patch_overlap)
                    rr = rl+self.patch_size
                    cl = j*(self.patch_size - self.patch_overlap)
                    cr = cl+self.patch_size
                    if img.dtype==np.uint8:
                        self.patches.append(img[rl:rr,cl:cr,:].astype(np.float32)/255.)
                    else:
                        self.patches.append(img[rl:rr,cl:cr,:].astype(np.float32))
        self.patches = np.asarray(self.patches,dtype=np.float32)
        
        # remove patches according to the max std of three color channels
        if self.std_thresh != None:
            ch_patches = self.patches.reshape([-1,self.patch_size**2,3])
            max_ch_std = np.std(ch_patches,axis=1).max(axis=1)
            self.std_criteria_idx = np.where(max_ch_std/max_ch_std.max() > self.std_thresh)[0]
            print('Selected %d of %d patches (%.2f percent)'%(
                self.std_criteria_idx.size,
                self.patches.shape[0],
                self.std_criteria_idx.size/self.patches.shape[0]*100.))
            self.patches = self.patches[self.std_criteria_idx,:,:,:]

        self.patches = self.patches.reshape(len(self.patches),3*(self.patch_size**2))
        self.raw_patches = self.patches
        if self.whiten:
            self.patches = whiten(self.patches)
        self.patches = torch.from_numpy(self.patches).to(self.device)


## ================================================================================
#  CONE NATURAL SCENES
## ================================================================================   
'''
class conenaturalscenes(Dataset,naturalscenes):
    
    def __init__(self,img_dir,patch_size,patch_overlap,data_key='IMAGESr',device=None):
        """
        Parameters:
        img_dir - string
            *.mat file to load images from, expected to be of for (pix,pix,n_img)
        patch_size - scalar (1,)
            patch row/column size to extract
        patch_overlap - scalar (1,) default=0
            amount to overlap patches by in pixels. If 0, no overlap
        data_key - string default='IMAGESr'
            key to query mat dict with to get data
        device - torch.device default == cpu
            device to load data on
        ---
        """
        # extract patches
        super().__init__(img_dir,patch_size,patch_overlap,data_key,device)

        # converrt patches back to numpy
        self.patches = torchtonumpy(self.patches)
'''

        
def whiten(data):
    """
    Whiten data via eigen decomposition
    ---
    parameters:
    img - scalar (N,n)
        input data where N is the number of points and n is features
    ---
    returns:
    wdata - scalar (N,n)
        whitened data
    """
    
    cdata = data.T-data.mean()
    
    cov = np.cov(cdata)
    w, v = np.linalg.eig(cov)
    diagw = np.diag(np.real(1/(w**0.5)))
    
    wdata = v@diagw@v.T@cdata
    return wdata.T.astype(np.float32)
    
    
def torchtonumpy(x):
    # convert pytorch x tensor to numpy array
    return x.cpu().detach().numpy()
    
        