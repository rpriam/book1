#from sklearn import datasets  
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import numpy as np 
import numpy.random as rd
import random as random

import torch.nn as nn
import torch
import tables
import h5py


import copy


from sklearn.metrics import accuracy_score,precision_score, \
                             recall_score,confusion_matrix
                             
from sklearn.metrics import mean_squared_error,r2_score

from sklearn import metrics

from sklearn.decomposition import IncrementalPCA

from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
import openTSNE

from scipy.special import factorial, gammaln

import time

import gc

#import matplotlib.pyplot as plt



########################################################

######################################################################
###### CHAPTER 01 ####################################################
######################################################################


    

######################################################################
###### CHAPTER 02 ####################################################
######################################################################



######################################################################
###### CHAPTER 03 ####################################################
######################################################################



######################################################################
###### CHAPTER 04 ####################################################
######################################################################

def f_train_glmr(dl_train,model,optimizer,loss,monitor,device=None,printed=1, 
                 loss_yy_model = None, transform_Xb=None,
                 is_transform_Xb_before_device = False,
                 transform_yb=None, transform_yhatb=None, 
                 update_model=None,):
    
    #print("device in glmr =", device)
    
    if device is not None: model.to(device)
    model.train()
    nbmax_epoqs    = monitor.nbmax_epoqs
    debug_out      = monitor.debug_out
    loss_train_s   = np.zeros(nbmax_epoqs)
    n_sample       = len(dl_train.dataset)
    monistop       = False
    t=0 #epoch
    while True:
        loss_epoch=0
        for b,tuple_b in enumerate(dl_train):
            Xb = tuple_b[0]
            yb = tuple_b[1]
            # data from mini batch into cpu or gpu
            if is_transform_Xb_before_device:
               if transform_Xb is not None: Xb = transform_Xb(Xb) 
            if device is not None: Xb = Xb.to(device)
            if device is not None: yb = yb.to(device)
            # eventually transform the data (if not in dataloader)
            if transform_yb is not None: 
                yb = transform_yb(yb,model,device)
            if not is_transform_Xb_before_device:
                if transform_Xb is not None: Xb = transform_Xb(Xb)
            def loss_closure():
                # predicted target from neural network
                yhatb = model(Xb)
                ###monitor.debug_var = yhatb
                # eventually transform the predicted variable (same shape than yb)
                if transform_yhatb is not None: yhatb = transform_yhatb(yhatb,model.name) 
                # gradient set to zero
                optimizer.zero_grad() 
                # loss computation from true and predicted targets
                #print(loss)
                #print(yhatb.shape, yb.shape)
                if loss_yy_model is None:
                    lossb = loss(yhatb, yb)
                else:
                    lossb = loss_yy_model(loss(yhatb, yb),model)
                # gradient numerical computation of loss + nn by backpropagation
                lossb.backward()                           # backward step
                return lossb
            #
            if isinstance(optimizer, torch.optim.LBFGS) is not True:
                lossb = loss_closure()
                if update_model==None:            
                    optimizer.step()                           # update of parameters
                else:
                    update_model(model,loss,optimizer,device,b,Xb,yb)   # own function for update of parameters (one worker) 
            #                                   # warning: usual pytorch function to be preferred
            else:
                optimizer.step(loss_closure)           # update of parameters
                lossb = loss_closure()
            #
            loss_epoch += lossb                        # loss accumulation
            #
            ### if ??? torch.cuda.empty_cache() 
            #Xb=Xb.cpu()
            #yb=yb.cpu()
            #del Xb, yb
            #gc.collect()
            #
        monitor.loss_train_s[t] = loss_epoch           # loss at epoch
        monistop = monitor.stop(t)                     # eventual stop
        # begin for printing on the computer terminal output for information
        if debug_out>0:
            debug_out=int(debug_out)
            if printed==1:
                if t%debug_out==0 or t==0:
                    print("loss=",round(monitor.loss_train_s[t],5), " \t t=",t,"/",
                          nbmax_epoqs," \t (",round(100*t/nbmax_epoqs,2),"%)")
            if printed==2 and t==0:
                    print("loss=",round(monitor.loss_train_s[t],5), " \t t=",t,"/",
                          nbmax_epoqs," \t (",round(100*t/nbmax_epoqs,2),"%)")
        # end for printing on the computer terminal output for information
        # increase epoch variable
        t=t+1 
        # check if loop must end when maximum number reach or stopping rule true
        if monistop or t==nbmax_epoqs:
            if printed==2:
                print("loss=",round(monitor.loss_train_s[(t-1)],5), " \t t=",(t),"/",
                      nbmax_epoqs," \t (",round(100*(t)/nbmax_epoqs,2),"%)")
            break
    #
    tmax= t
    return monitor.loss_train_s, tmax, monistop


######################################################################
###### CHAPTER 05 ####################################################
######################################################################

    
def f_normalizeData(x_train,x_test):
    meanx_train = np.mean(x_train,axis=0)
    sdx_train   = np.sqrt(np.var(x_train,axis=0))
    sdx_train [sdx_train==0] = 1
    
    x_train = (x_train-meanx_train)/sdx_train
    x_test  = (x_test-meanx_train)/sdx_train
    return x_train, x_test
    
    
def f_idx_traintest_kfolds(n,k_fold=5,shuffle = True):
    if not shuffle : idx_all = range(n)
    if shuffle     : idx_all = rd.permutation(range(n))
    idx_s = dict()
    for k,idx_test in enumerate(np.array_split(idx_all,k_fold)):
        idx_train = [e for e in idx_all if e not in idx_test]
        idx_s[str(k)] = dict({"train":np.asarray(idx_train),
                              "test":np.asarray(idx_test)})
    return idx_s
    


#optimizer implemented is sgd() one
def f_reg_l1_nn_cv(idx_s,namefile_s,dataset,model_,loss_,batch_size,alpha_t=1e-5,
                   nbmax_epoqs=5000, debug_out=50, device=None, 
                   transform_yb=None, transform_yhatb=None, transform_Xb=None,
                   loss_yy_model=None, printed=2, hyperparameter_to_print=None):
    loss_train_s_s                                 = []
    loss_test_s_s                                  = []
    yhat_train_s                                   = []
    y_train_s                                      = [] 
    yhat_test_s                                    = [] 
    y_test_s                                       = []
    
    if dataset is not None: dict_s = idx_s
    if dataset is None: dict_s = namefile_s
    
    for k, key_k in enumerate(dict_s.keys()):    
        if printed>0: print("processing fold n° "+str(k+1),"/",str(len(dict_s)))
        
        if dataset is not None:
            idx_train, idx_test = idx_s[key_k]["train"], idx_s[key_k]["test"]
            n_train, n_test     = len(idx_train), len(idx_test)
            # print(n_train, n_test)

            subsampler_train = torch.utils.data.SubsetRandomSampler(idx_train)
            subsampler_test  = torch.utils.data.SubsetRandomSampler(idx_test)

            dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                      sampler=subsampler_train)

            dl_test  = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                                     sampler=subsampler_test)
        else: #dataset is None
            namefile_train = dict_s[key_k]["train"]
            #print(namefile_train)
            dataset_train = DatasetH5(namefile_train,'x','y',getindex=False) 
            dl_train      = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
            
            namefile_test = dict_s[key_k]["test"]
            #print(namefile_test)
            dataset_test = uDatasetH5(namefile_test,'x','y',getindex=False) 
            dl_test      = torch.utils.data.DataLoader(dataset_test,batch_size=batch_size)
                        
        model          = copy.deepcopy(model_)
        loss           = loss_
        optimizer      = torch.optim.SGD(model.parameters(), lr=alpha_t, momentum=0.0)
        monitor        = MyMonitorTest(model,loss,dl_train,dl_test,
                                             nbmax_epoqs,debug_out,device,
                                             transform_yb=transform_yb,
                                             transform_xb=transform_Xb)
        
        loss_train_s,tmax,monistop = \
        f_train_glmr(dl_train,model,optimizer,loss,monitor,device,printed,
                           transform_yb=transform_yb, transform_yhatb=transform_yhatb,
                           transform_Xb=transform_Xb,loss_yy_model=loss_yy_model)
        
        yhat_train, y_train = f_get_yhat(model,dl_train,device=device)
        yhat_test, y_test   = f_get_yhat(model,dl_test,device=device)

        if printed>1:
            print("hyperparameter:",str(hyperparameter_to_print),"alpha_t=",str(np.round(alpha_t,5)), 
                  " mse_train=",str(np.round(np.mean( (y_train-yhat_train)**2 ),3)),
                  " mse_test=",str(np.round(np.mean( (y_test-yhat_test)**2 ),3)))
        
        loss_train_s_s.append(loss_train_s)
        loss_test_s_s.append(monitor.loss_test_s)
        yhat_train_s.append(yhat_train)
        y_train_s.append(y_train)
        yhat_test_s.append(yhat_test)
        y_test_s.append(y_test)
    
    return  loss_train_s_s, loss_test_s_s, yhat_train_s, \
            y_train_s, yhat_test_s, y_test_s
            
            

class DatasetH5(torch.utils.data.Dataset):
    def __init__(self, file_path, xname='x', yname='y',getindex=True):
        super(DatasetH5, self).__init__()
        h5_file = h5py.File(file_path , 'r')
        self.x = h5_file[xname]
        self.y = h5_file[yname]
        self.getindex = getindex
#
    def __getitem__(self, index): 
#         return (torch.from_numpy(self.x[index,:]).float(),
#                 torch.from_numpy(self.y[index,:]).int(),
#                 index)
        if self.getindex:
            return (np.float32(self.x[index,:]),
                    np.float32(self.y[index,:]),
                    np.int32(index))
        else:
            return (np.float32(self.x[index,:]),
                    np.float32(self.y[index,:]))
#
    def __len__(self):
        return self.x.shape[0]


######################################################################
###### CHAPTER 06 ####################################################
######################################################################



def f_varianceMatrixFromGradients_ForParameters_modelNN(model, loss, dataloader, n_train,device=None):
    p_model = f_get_p_model(model)
    Iapprox = torch.zeros((p_model,p_model))
    if device is not None: Iapprox = Iapprox.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.0)
    optimizer.zero_grad()

    for b,(Xb,yb) in enumerate(dataloader):
        if device is not None:
            Xb = Xb.to(device)
            yb = yb.to(device)
        #print(".", end = '')
        for i in range(Xb.shape[0]):  # to change into matrix compute (avec matrix diag)
            Xb_i = Xb[i,:]
            yb_i = yb[i].ravel()
            yhatb_i = model(Xb_i).ravel()
            loss_b = loss(yhatb_i, yb_i)
            optimizer.zero_grad()
            loss_b.backward()
            gradient_vect = []
            with torch.no_grad():
                for p in model.parameters():
                    gradient_vect.append(p.grad.view(-1))
                gradient_vect = torch.cat(gradient_vect)
                gradient_vect = gradient_vect.reshape((p_model,1))
                Iapprox = Iapprox + gradient_vect @ gradient_vect.T /n_train
    return Iapprox, p_model
    
    
def f_varianceMatrixFromFullHessian_ForParameters_modelNN(model, loss, 
                                                          dataloader,n_train,
                                                          loss_yy_model = None,
                                                          device=None):
    
    #print("device in hessian =", device)
    
    p_model = f_get_p_model(model)
    Imodel  = torch.zeros((p_model,p_model)) 
    if device is not None: 
        Imodel = Imodel.to(device)
        model = model.to(device)
    
    #if device is not None: model = model.to(device)
    model.eval()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.0)
    optimizer.zero_grad()
    
    for b,(Xb,yb) in enumerate(dataloader):
        #print(".", end = '')
        
        if device is not None: Xb=Xb.to(device)
        if device is not None: yb=yb.to(device)
            
        yhatb = model(Xb)
        if device is not None: yhatb = yhatb.to(device) # transformation ?????? <- linear & logit ! 
        
        #loss_b      = loss(yhatb, yb.reshape(yhatb.shape))
        # if loss_yy_model is None:
    
        #yb = transform_yb(yb, model, device)
            
        loss_b = loss(yhatb, yb.reshape(yhatb.shape))
        # else:
        # loss_b = loss_yy_model(loss(yhatb, yb.reshape(yhatb.shape)),model)
               
        grad1rds_list = torch.autograd.grad(loss_b, model.parameters(), \
                                         create_graph=True, \
                                         retain_graph=True, \
                                         allow_unused=True)

        grad1rds_vec = torch.cat([g.view(-1) for g in grad1rds_list]) #.squeeze()
        
        grad2rds_list = []
        for grad1rd in  grad1rds_vec:
            grad2rds_1row = torch.autograd.grad(grad1rd, model.parameters(), \
                                          create_graph=True, \
                                          retain_graph=True, \
                                          allow_unused=True)

            grad2rds_1vect = torch.cat([g.view(-1) for g in grad2rds_1row]) #.squeeze()

            grad2rds_list.append( grad2rds_1vect )

        for k in range(p_model):
            Imodel[k,:] += grad2rds_list[k] / n_train
        
    return Imodel, p_model


######################################################################
###### CHAPTER 07 ####################################################
######################################################################

def f_hessian_poisson_newton(X_,y_,mu_,g_new,H_old):    
    return  -(X_.T @ (mu_ * X_)) # hessian

def f_gradient_poisson(X_,y_,mu_):
    return X_.T @ (y_ - mu_)    # gradient


def f_save_dl_xy_to_2memmap(dataloader,
                            filename_x, 
                            filename_y,
                            n=None,p=None,
                            is_index=False,
                            x_transform=None):
    if n==None or p==None:
        n, p  = dataloader.dataset.tensors[0].shape
    
    x_map = np.memmap(filename_x, \
                         dtype='float32', mode='w+', shape=(n,p))
    
    y_map = np.memmap(filename_y, \
                         dtype='float32', mode='w+', shape=(n,1))
    i_b=0
    if is_index:
        for b,(xb,yb,ib) in enumerate(dataloader):
            if x_transform is not None: xb=x_transform(xb)
            x_map[i_b:(i_b+len(yb)),:] = xb
            y_map[i_b:(i_b+len(yb)),:] = yb.reshape((len(yb),1))
            i_b = i_b + len(yb)
    else:
        for b,(xb,yb) in enumerate(dataloader):
            if x_transform is not None: xb=x_transform(xb)
            x_map[i_b:(i_b+len(yb)),:] = xb
            y_map[i_b:(i_b+len(yb)),:] = yb.reshape((len(yb),1))
            i_b = i_b + len(yb)

    np.savetxt(str(filename_x+".shape.txt"),[n,p])
    np.savetxt(str(filename_y+".shape.txt"),[n,1])

    del x_map, y_map
    
    

def f_train_my_glm(dl_train, dl_test, layers, name_model,                   
                   nbmax_epoqs=10, debug_out=1, alpha_t=0.001,
                   device=None, momentum = 0.0, init_model = None,
                   transform_yb = None, transform_yhatb = None,
                   transform_xb = None, loss=None,
                   is_transform_Xb_before_device = False,
                   update_model=None, printed=0,
                   reduction="sum",loss_yy_model=None,
                   name_optimizer = "SGD", nbmax_iter_lbgs = 8,
                   K=None,init_w=True):   #, model=None):
    
    # select loss
    if loss==None:
        if (name_model== "LinearRegression" or \
            name_model== "MLP"):
                loss = torch.nn.MSELoss(reduction=reduction)
        
        if (name_model== "LogisticRegression" or \
            name_model== "LMLP"):
                loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        
        if (name_model== "SoftmaxRegression" or \
            name_model== "MultinomialRegression" or \
            name_model== "SMLP" or \
            name_model== "MMLP"):
                loss = torch.nn.CrossEntropyLoss(reduction=reduction)
                
#         if (name_model== "MultiLogitRegression" or \
#             name_model== "MLMLP"):
#                 loss = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        
        if (name_model== "PoissonRegression" or \
            name_model== "PMLP"):
            #loss = poisson_cross_entropy
            loss = torch.nn.PoissonNLLLoss(reduction=reduction, \
                                      log_input=True, full=True, )
    
    if loss==None: loss = torch.nn.MSELoss(reduction=reduction)
    
    #check model names
    if (name_model!= "LinearRegression" and \
        name_model!= "LogisticRegression" and \
        name_model!= "SoftmaxRegression" and \
        name_model!= "MultinomialRegression" and \
        name_model!= "PoissonRegression" and \
#         name_model!= "MultiLogitRegression" and \
        name_model!= "MLP" and \
        name_model!= "LMLP" and \
        name_model!= "SMLP" and \
        name_model!= "MMLP" and \
#         name_model!= "MLMLP" and \
        name_model!= "PMLP"): name_model = "MLP"
    
    #if model is None:
    model     = GNLMRegression(name_model,copy.deepcopy(layers),init_w=init_w)
    
    if (name_model== "SoftmaxRegression" \
            or name_model== "MultinomialRegression" \
            or name_model== "SMLP" \
            or name_model== "MMLP"):# \
            #or name_model== "MultiLogitRegression" \
            #or name_model== "MLMLP"):
                model.K=K ####if model.K is None: 

    if name_optimizer!= "SGD" and name_optimizer!= "LBFGS": optimizer = None
    
    if name_optimizer== "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=alpha_t, \
                                    momentum=momentum)
    
    if name_optimizer== "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=alpha_t, \
                                      max_iter=nbmax_iter_lbgs)
    
    monitor   = MyMonitorTest(model,loss,dl_train,dl_test,
                                    nbmax_epoqs,debug_out,device,
                                    transform_yb = transform_yb,                                    
                                    transform_xb = transform_xb)
    # train model
    #if device is not None: model=model.to(device)
            
    #model.train()
    loss_s,tmax,monistop = \
              f_train_glmr(dl_train,model,optimizer,loss,monitor, 
                          device=device, 
                          transform_yb = transform_yb,
                          transform_yhatb = transform_yhatb,
                          transform_Xb = transform_xb,
                          is_transform_Xb_before_device = \
                            is_transform_Xb_before_device,
                          update_model=update_model,
                          printed=printed,
                          loss_yy_model=loss_yy_model)
    
    return {"loss_train_s":loss_s, "tmax":tmax, "monistop":monistop,
           "model":model,"monitor":monitor, "loss":loss,
           "dl_train":dl_train, "dl_test":dl_test}


######################################################################
###### CHAPTER 08 ####################################################
######################################################################


def f_plot_scatter(z,y,title="",xlabel="",ylabel="",isellipse=False):
    y = y.astype(np.int8).ravel()
    
    if isellipse is False:
        fig = plt.figure(figsize=(8, 6), dpi=80)
        fig.set_figwidth(8)
        fig.set_figheight(6)
        plt.title(title)
        for k in iter(np.unique(y)):
            # plt.scatter(z[:,0], z[:,1], c=np.array(cols)[y])
            plt.scatter(z[y==k,0], z[y==k,1], c=cols[k], s=1.5) #cmap='tab10'
        plt.gca().set(#aspect='equal',
                      title=title,
                      xlabel=xlabel, ylabel=xlabel)
    
    if isellipse is True:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
        fig.set_figwidth(8)
        fig.set_figheight(6)
        plt.title(title)
        for k in iter(np.unique(y)):

            x1_s   = z[y == k, 0]
            x2_s   = z[y == k, 1]
            y_s    = y[y == k]

            cov = np.cov(x1_s, x2_s)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:,order]
            theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
            w, h = 3.5 * np.sqrt(vals)

            ell = Ellipse(xy=(np.mean(x1_s), np.mean(x2_s)), width=w, height=h,
                          angle=theta, color='black', alpha=0.1)
            ell.set_facecolor(cols[k])
            ax.add_artist(ell)

            ell2 = Ellipse(xy=(np.mean(x1_s), np.mean(x2_s)), width=w, height=h,
                           angle=theta, color='black', alpha=1)
            ell2.set_facecolor('None')
            ax.add_artist(ell2)
    
            #ax.scatter(x1_s, x2_s, label='.', c=y_s, cmap='tab10', lw = 0, alpha=1, s=1.5) 
            ax.scatter(x1_s, x2_s, label='.', c=cols[k], lw = 0, alpha=1, s=1.5) 
        
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
    
    plt.show()
    
    return plt



# save dataloader into h5py file
# x,y from dataloader et nb size of a minibatch
# with y labels: y(1,d)
# with x data  : array xb(nb,d) or xb(nb,1,d1,d2) with d=d1*d2
# such that nb = x.shape[0] -> reshape x(nb,d) in file            
def f_save_dataloader_to_h5py(dataloader,filename,d,transformx=None):  
#     for checking d, not implemented
#     (xb, yb) = next(iter(dataloader))
#     xb = xb.detach().numpy()
#     yb = yb.detach().numpy()
    file = tables.open_file(filename, mode='w')
    
    x_atom = tables.Float64Atom()
    y_atom = tables.Int16Atom()
    x_ds = file.create_earray(file.root, 'x', x_atom,(0,d))
    y_ds = file.create_earray(file.root, 'y', y_atom,(0,1))
    
    for step, tuple_b in enumerate(dataloader):
        xb = tuple_b[0]
        yb = tuple_b[1]
        xb = xb.detach().numpy()
        if transformx is not None:
            xb = transformx(xb)
        xb = xb.reshape((len(xb),d))
        yb = yb.detach().numpy()
        yb = yb.reshape((len(yb),1))
        x_ds.append(xb)
        y_ds.append(yb)
    
    file.close()
    
    

def f_save_cvdatasets_to_h5py(idx_s,dataset,filename_base,
                              d_aftertransformx,transformx=None,
                              show=0):  
    
    namefile_s = dict()
    
    for k, key_k in enumerate(idx_s.keys()):    
        if show: print("processing fold n° "+str(k+1),"/",str(len(idx_s)))
        
        idx_train, idx_test = idx_s[key_k]["train"], idx_s[key_k]["test"]
        n_train, n_test     = len(idx_train), len(idx_test)
        # print(n_train, n_test)
        
        subsampler_train = torch.utils.data.SubsetRandomSampler(idx_train)
        subsampler_test  = torch.utils.data.SubsetRandomSampler(idx_test)

        dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                  sampler=subsampler_train)

        dl_test  = torch.utils.data.DataLoader(dataset,batch_size=batch_size,
                                                 sampler=subsampler_test)

        filename_train = filename_base+str("_")+str(k)+str("_")+"train.h5"
        filename_test = filename_base+str("_")+str(k)+str("_")+"test.h5"
    
        f_save_dataloader_to_h5py(dl_train,filename_train,d_aftertransformx,transformx)
        f_save_dataloader_to_h5py(dl_test,filename_test,d_aftertransformx,transformx)
    
        namefile_s[str(k)] = dict({"train":filename_train,
                                   "test":filename_test})
    
    return  namefile_s


def f_save_standardized_memmap2memmap(xin_memmap,filame_xout_memmap,
                                      size_chunks = 100):
    mn = np.sum(xin_memmap,axis=0)          # to compute by chunks ?
    sd = np.sqrt(np.var(xin_memmap,axis=0)) # to compute by chunks ?
    sd[sd==0] = 1
    n = xin_memmap.shape[0]
    
    xout_memmap_strd = np.memmap(filame_xout_memmap, 
                                dtype='float32', mode='w+', 
                                shape=(xin_memmap.shape[0],
                                       xin_memmap.shape[1]))

    for idx_b in range(0, n, size_chunks):
        idx_b2 = np.min( [idx_b+size_chunks,n] )
        zb     = xin_memmap[idx_b:idx_b2,:]
        xout_memmap_strd[idx_b:idx_b2,:] = (zb - mn)/sd

    del xout_memmap_strd
    return mn, sd

######################################################################
###### CHAPTER 09 ####################################################
######################################################################

def f_plot_scatter3d(z,y,title=" ",xlabel="x",ylabel="y",s=1.5):
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.axes(projection='3d')
    for k in iter(np.unique(y)):
        ax.scatter3D(z[:,0], z[:,1], z[:,2], c="black",s=s)
    plt.gca().set(#aspect='equal',
                  title=title,
                  xlabel=xlabel, ylabel=ylabel)
    plt.show()

def f_print_accuracies(model, dl_train, dl_test,printed=True):  
    acc_train, _, _ = f_test_glmr(model.cpu(),dl_train, True)
    acc_test, _, _  = f_test_glmr(model.cpu(),dl_test, True)
    if printed: print(f"acc_train = {acc_train:1.4f}",
                      f" acc_test  = {acc_test:1.4f}")
    return acc_train, acc_test


def f_plot_losses(monitor,n_train,n_test):
        loss_train_s = monitor.loss_train_s / n_train
        loss_test_s = monitor.loss_test_s[monitor.loss_test_s>0] / n_test
        t_train_s = range(len(loss_train_s))
        t_test_s = monitor.step_test_s[monitor.loss_test_s>0].astype(int)
        plt = f_draw_s([ t_train_s , t_test_s ],
                       [ loss_train_s, loss_test_s],
                       ["b-", "r-"] ,"t",[ "loss train", "loss test"], " ")


def f_meanstd_from_dataloader(dataloader,device):
    
    #find number of columns
    xb1, yb1, ib1 = next(iter(dataloader))
    p_train = xb1.shape[1]
    
    #find number of rows
    n_train = 0
    for b, (xb, yb, ib) in enumerate(dataloader):
        n_train += xb.shape[0]
    
    #init mean, std
    mean_x_train = torch.zeros((1,p_train))
    std_x_train = torch.zeros((1,p_train))
    
    #compute mean, std from minibatches
    for b, (xb, yb, ib) in enumerate(dataloader):
        mean_x_train += torch.sum(xb, axis=0).reshape((1,p_train)) / n_train
        std_x_train  += torch.sum(xb**2, axis=0).reshape((1,p_train)) / n_train
    std_x_train = torch.sqrt(std_x_train - mean_x_train**2)
    
    #check if some components are zero in std and fix
    bool_std_x_train_wasnull = (std_x_train==0).detach().numpy()
    std_x_train[std_x_train==0] = 1 # for pixel always zero or column constant

    #tensor in device (cpu or gpu)
    mean_x_train = mean_x_train.to(device)
    std_x_train = std_x_train.to(device)

    return mean_x_train, std_x_train, bool_std_x_train_wasnull


