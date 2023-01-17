# python code for the book "Linear and Deep Models Basics with Pytorch, Numpy, and Scikit-Learn" (december 2022)


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

cols = ["blue","red","green","orange","purple","brown","olive","magenta","cyan", "black"]

def towdir(s):
    return (str('./datasets_book/'+s))

########################################################

######################################################################
###### CHAPTER 01 ####################################################
######################################################################

def fun_plot_polynom(x,y_true,y_hat_s,degree_s,showlegend=True,namesample="whole",draw=True):
    if draw: plt.scatter(x, y_true, color='black',marker='.', s=50)
    if draw: axis = plt.axis()
    err_s = np.zeros((len(degree_s),))
    idx_sort = np.argsort(x.ravel())
    for c, (y_hat,degree) in enumerate(zip(y_hat_s,degree_s)):
        y_hat = y_hat[0]
        y_hat = y_hat.ravel()
        mse = np.sum((y_hat[idx_sort]-y_true[idx_sort])**2,axis=0)/len(y_hat)
        #print(degree, distance_true_fitted)
        if draw:
            plt.plot(x[idx_sort], y_hat[idx_sort], 
                     label='degree={0}'.format(degree))
            plt.xlim(-1.5, 1.5)
            plt.ylim(-4.5, 7.5)
            plt.title("Error for the "+str(namesample)+" sample")
            if showlegend:
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        err_s[c] = mse
    return err_s
    

######################################################################
###### CHAPTER 02 ####################################################
######################################################################

def f_sigmoid(a):
    return np.exp(a)/(1+np.exp(a))
    

def nprd(x,ndec=3): return np.round(x,ndec)


def f_draw(x,y,markercolor,xlabel,ylabel,title,ax):
    
    ax.plot(x,y, markercolor)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def f_vizu2d_beta(ax,x1_yeql1,x1_yeql2,x2_yeql1,x2_yeql2,
                  beta_list, marker_list, xlim=None, ylim=None,
                  samplename="Sample"):
    x1 = np.append(x1_yeql1,x1_yeql2)
    x2 = np.append(x2_yeql1,x2_yeql2)
    x1_min=min(x1)
    x1_max=max(x1)
    
    ax.plot(x1_yeql1, x2_yeql1, 'bx', x1_yeql2, x2_yeql2, 'bo')
    
    for beta, marker in iter(zip(beta_list,marker_list)):
        x1_ = sigma_grid   = np.mgrid[x1_min:x1_max:1000j]
        x2_ = -(beta[0]+x1_*beta[1])/beta[2]        
        plt.plot(x1_,x2_,marker)

    if xlim is not None: ax.set_xlim(xlim)    
    if ylim is not None: ax.set_ylim(ylim)
    
    ax.set(xlabel=r'$x1$', ylabel=r'$x2$')    
    # plt.xlabel(r'$x1$')   
    # plt.ylabel(r'$x2$')
    
    # plt.title(r'Sample points and separation(s) line(s)')  
    ax.set_title(r'Sample points and separation(s) line(s)')
    ax.set_title(samplename + " points and separation(s) line(s)")
    # plt.show()

def f_metrics_regression(y,yhat,printed=False,drawn=False,ax=None,ndec=2,samplename="sample"):
#     np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    
    mse_yyhat = mean_squared_error(y.ravel(), yhat.ravel())
    r2_yyhat  = r2_score(y.ravel(), yhat.ravel())
    #cor_yyhat = np.corrcoef(y.ravel(),yhat.ravel())[0,1]
    
    if printed:
        print("MSE = ", nprd(mse_yyhat,ndec), "\n"
              "R2  = ", nprd(r2_yyhat,ndec),) 
              #"COR_train : ", "{:.3f}".format(cor_yyhat))
    
    if drawn==True:        
        f_draw(y,yhat,"b.","y (true target)","y_hat (predicted target)",
               str("Scatterplot for regression from "+samplename),ax)
        
    return mse_yyhat,r2_yyhat  #,cor_hat
    

def f_metrics_classification(y,yhat,printed=False,drawn=False,ax=None,ndec=3,samplename="sample"):
    cm_yyhat = metrics.confusion_matrix(y, yhat)
    acc_yyhat = metrics.accuracy_score(y, yhat)
    prc_yyhat = metrics.precision_score(y, yhat)
    rcc_yyhat = metrics.recall_score(y, yhat)
    if printed:
        print("Confusion matrix")
        print(cm_yyhat)
        print("Accuracy  = ", nprd(acc_yyhat,ndec))
        print("Precision = ", nprd(prc_yyhat,ndec))
        print("Recall    = ", nprd(rcc_yyhat,ndec))
    if drawn:
        pass
    
    return acc_yyhat, prc_yyhat, rcc_yyhat, cm_yyhat

######################################################################
###### CHAPTER 03 ####################################################
######################################################################



######################################################################
###### CHAPTER 04 ####################################################
######################################################################

def f_splitIndex(n,percentage=0.25):
    n_test         = int(n*percentage)
    n_train        = int(n-n_test)

    ids_all        = range(n)
    
    ids_test       = random.sample(ids_all,n_test)
    ids_train      = list(set(ids_all)-set(ids_test))
    
    ids_train = np.random.permutation(ids_train)
    ids_test = np.random.permutation(ids_test)
    
    return ids_train, ids_test, ids_all
    
   
def f_splitData(x,y,percentage=0.2):
    ids_train, ids_test, ids_all = f_splitIndex(len(y),percentage)
    x_train = x[ids_train,:]
    y_train = y[ids_train]
    x_test = x[ids_test,:]
    y_test = y[ids_test]
    meanx_train = np.mean(x_train)
    sdx_train   = np.sqrt(np.var(x_train))
    x_train     = (x_train-meanx_train)/sdx_train
    x_test      = (x_test-meanx_train)/sdx_train
    return x_train, x_test, y_train, y_test
    
def f_splitDataset(dataset,percentage=0.8,batch_size=10,shuffle=False):  #0.08?
    n = len(dataset)
    n_train = int(percentage * n)
    n_test = n- n_train
    dataset_train, dataset_test = torch.utils.data.random_split(dataset,  [n_train, n_test])
    #not normalized here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dl_train  = DataLoader(dataset_train, batch_size= batch_size, shuffle=shuffle)
    dl_test  = DataLoader(dataset_test, batch_size= batch_size, shuffle=False)
    return dl_train, dl_test, n, n_train, n_test  
    
def f_splitDataToStandadizeDL(x,y,percentage=0.2,batch_size=10,standardize=False,shuffle=False):

    x_train, x_test, y_train, y_test = f_splitData(x,y,percentage=percentage)
    
    if standardize:
        mx=np.mean(x_train,axis=0)
        stdx=np.sqrt(np.var(x_train,axis=0))
        stdx[stdx==0] = 1 #columns with constants
        x_train = (x_train-mx)/stdx
        x_test = (x_test-mx)/stdx
    
    dt_train   = TensorDataset( torch.from_numpy(x_train).float(), 
                                    torch.from_numpy(y_train).int() )

    dl_train  = DataLoader(dt_train, batch_size= batch_size, shuffle=shuffle)

    dt_test   = TensorDataset( torch.from_numpy(x_test).float(), 
                                    torch.from_numpy(y_test).int() )

    dl_test  = DataLoader(dt_test, batch_size= batch_size, shuffle=False)
    
    n_train = x_train.shape[0]
    n_test  = x_test.shape[0]
    p       = x.shape[1]
    
    return dl_train, dl_test, n_train, n_test, p


class GNLMRegression(nn.Module):
    def __init__(self, name, layers, K=None,init_w = True):
        super().__init__()
        self.name = name
        self.K    = K
        self.layers = layers
        self.net = nn.Sequential(*layers)
        if init_w == True: torch.nn.init.xavier_uniform_(self.net[0].weight)
        #torch.nn.init.xavier_uniform_(self.net[1].weight)
    def forward(self, x):
        return self.net(x)


def extract_weights_lin(model):
    weight = bias = None
    for param_tensor in model.state_dict():
        if (param_tensor=="lin.weight"):
            weight = model.state_dict()[param_tensor]
        if (param_tensor=="lin.bias"):
            bias = model.state_dict()[param_tensor]    
    return np.append(bias, weight)


def transform_yb(yb,model,device=None):
    name_model = model.name
    
    if (name_model== "SoftmaxRegression" or \
            name_model== "MultinomialRegression" or \
            name_model== "SMLP" or \
            name_model== "MMLP"):
        yb = yb.long() # type needs to be changed for some losses
        
    if name_model == "LinearRegression"  or name_model == "MLP" \
        or name_model == "LogisticRegression" or name_model == "LMLP" \
        or name_model == "PoissonRegression" or name_model == "PMLP" \
        or name_model== "SoftmaxRegression" or name_model == "SMLP" \
        or name_model == "MultinomialRegression" or name_model== "MMLP":
        yb = yb.ravel()
    
#     if name_model == "MultiLogitRegression" or name_model == "MLMLP":
#         ze = torch.zeros(len(yb), model.K) 
#         if device is not None: ze = ze.to(device)
#         yb =yb.to(torch.long)
#         ze[range(ze.shape[0]), yb]=1.0
#         yb=ze
    
    return yb


def transform_yhatb(yhatb,name_model):
    if name_model == "LinearRegression"  or name_model == "MLP" \
        or name_model == "LogisticRegression" or name_model == "LMLP" \
        or name_model == "PoissonRegression" or name_model == "PMLP": 
        yhatb = yhatb.ravel()
    return yhatb
    

class MyMonitor: ## to do: ajouter tensorboard, ajouter accuracy
    def __init__(self,model,dataloader_train=None,nbmax_epoqs=1e4,device=None):
        self.model            = model
        self.dataloader_train = dataloader_train
        self.nbmax_epoqs      = nbmax_epoqs
        self.device           = device
    def stop(self,t,loss_train=None):                  # not implemented yet
        return False

class MyMonitorTest(MyMonitor):
    def __init__(self,model,loss,dataloader_train=None,dataloader_test=None,
                 nbmax_epoqs=1e4,debug_out=1e2,device=None,
                 transform_yb=None,transform_xb=None):
        super().__init__(model,dataloader_train,nbmax_epoqs,device)
        self.loss            = loss
        self.dataloader_test = dataloader_test
        self.loss_train_s    = np.zeros(nbmax_epoqs)  # the loss for the train set
        self.loss_test_s     = np.zeros(nbmax_epoqs)  # the loss for the test set
        self.step_test_s     = np.zeros(nbmax_epoqs)  # the steps for the test loss
        self.debug_out       = debug_out
        self.transform_yb    = transform_yb
        self.transform_xb    = transform_xb
    def stop(self,t):                        # partially implemented
        if t%int(self.debug_out)==0 or t==0:
            with torch.no_grad(): 
                for tuple_b in iter(self.dataloader_test):
                    Xb = tuple_b[0]
                    yb = tuple_b[1] 
                    if self.device is not None:
                        yb = yb.to(self.device)
                        Xb = Xb.to(self.device)
                        if self.transform_xb is not None:
                            Xb=self.transform_xb(Xb)
                    yhatb = self.model(Xb)
                    if self.transform_yb is None:
                        yb = transform_yb(yb,self.model,self.device)
                    else:
                        yb = self.transform_yb(yb,self.model,self.device)
                    #if self.device is not None:
                     #   yb = yb.to(self.device)
                     #   Xb = Xb.to(self.device)
                    #yhatb = self.model(Xb)       
                    yhatb = transform_yhatb(yhatb,self.model.name)
                    loss_b = self.loss(yhatb, yb)
                    #torch.cuda.empty_cache()
                    self.loss_test_s[t] += loss_b        # loss accumulation
                    self.step_test_s[t] = t             # loss accumulation
                    #gc.collect()
                    #-- Garbage collector
                    ### if ??? torch.cuda.empty_cache() 
                    #Xb=Xb.cpu()
                    #yb=yb.cpu()
                    #del Xb, yb
                    #gc.collect()
                    #-- Acuracy
                    
                    # --
        return False
        

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


# def f_draw_s(x_list,y_list,markercolor_list,xlabel,ylabel_list,title,
#              plt=None,ishow=True,legendloc="best", legendsize=20):
#     if plt is None: import matplotlib.pyplot as plt
#     
#     for (x,y,markercolor,ylabel) in iter(zip(x_list,y_list,markercolor_list,ylabel_list)):
#         plt.plot(x,y, markercolor,label=ylabel)
#     
#     plt.title(title)    
#     plt.xlabel(xlabel)
#     # plt.ylabel(ylabel)
#     plt.legend(loc=legendloc, prop={'size': legendsize})
#     
#     if ishow is True:
#         plt.show()
#     
#     return plt
def f_draw_s(x_list,y_list,markercolor_list,xlabel,ylabel_list,title,
             ax,ishow=True,legendloc="best", legendsize=20):
    
    for (x,y,markercolor,ylabel) in iter(zip(x_list,y_list,markercolor_list,ylabel_list)):
        ax.plot(x,y, markercolor,label=ylabel)
    
    ax.set_title(title)    
    ax.set_xlabel(xlabel)
    # plt.ylabel(ylabel)
    ax.legend(loc=legendloc, prop={'size': legendsize})


def f_test_glmr(model,dataloader,is_yhat=False,threshold=0.5):
    stm = torch.nn.Softmax(dim=1)
    yhat = None
    y = None
    acc = 0
    with torch.no_grad(): 
        for tuple_b in iter(dataloader):
            Xb = tuple_b[0]
            yb = tuple_b[1]
            # Forward pass
            yhatb = model(Xb)
            if model.name == "LinearRegression":
                acc += torch.sum((yhatb == yb).float())
            elif model.name == "LogisticRegression" or model.name == "LMLP":
                yhatb = (torch.exp(yhatb)/(1+torch.exp(yhatb))>threshold).float()
                yb    = yb.reshape((len(yb),1))
                acc += torch.sum((yhatb.ravel() == yb.ravel()).float())
            elif model.name == "MultinomialRegression" \
                  or model.name == "SoftmaxRegression" or model.name == "SMLP":
                yhatb = stm(yhatb)
                yhatb = torch.argmax(yhatb, dim=1)
                acc += torch.sum((yhatb.ravel() == yb.ravel()).float())
            elif model.name == "MultinomialRegression_" \
                  or model.name == "SoftmaxRegression_" or model.name == "SMLP_":
                yhatb = stm(yhatb)
                yhatb = torch.argmax(yhatb, dim=1)
                acc += torch.sum((yhatb.ravel() == yb.ravel()).float())
                 #pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
                 #correct += pred.eq(ts.view_as(pred)).sum().item()
            if is_yhat:
                if yhat is None:
                    yhat = yhatb
                    y = yb
                else:
                    yhat = np.append(yhat,yhatb)
                    y    = np.append(y,yb)

    if is_yhat:
        yhat = yhat.squeeze()
        y = y.squeeze()
    
    return (float)(acc/len(dataloader.dataset)), yhat, y


def f_plot_2d_boudary_MLP(ax,model,x,y,nbs=100,threshold=0.5):
    stm = torch.nn.Softmax(dim=1)
    
    x1_s =  np.mgrid[min(x[:,0]):max(x[:,0]):complex(real=0, imag=nbs)]
    x2_s =  np.mgrid[min(x[:,1]):max(x[:,1]):complex(real=0, imag=nbs)]
    x1_x2_s = np.zeros((nbs*nbs,2))
    k=0
    for i in range(nbs):
        for j in range(nbs):
            x1_x2_s[k,:] = (x1_s[i],x2_s[j])
            k+=1
    x1_x2_s = torch.Tensor(x1_x2_s)
    with torch.no_grad():
        y12_s = model(x1_x2_s)
        if model.name == "MultinomialRegression" \
             or model.name == "SoftmaxRegression" \
             or model.name == "SMLP" \
             or model.name == "MMLP":
            y12_s = stm(y12_s)
            y12_s = torch.argmax(y12_s,axis=1)
        if model.name == "LogisticRegression" or model.name == "LMLP":
            y12_s = transform_yhatb(y12_s,model.name)
            y12_s = torch.exp(y12_s) / (1+torch.exp(y12_s))
            y12_s = (y12_s > threshold).int().detach().numpy() + 1
    #plt.figure()#figsize=(9,9))  
    ax.scatter(x1_x2_s[:,0], x1_x2_s[:,1],c=y12_s[:] , cmap=cm.Accent)
    ax.scatter(x[:,0], x[:,1],c=y[:] , cmap=cm.autumn, s=0.9)
    #plt.show()

######################################################################
###### CHAPTER 05 ####################################################
######################################################################

def f_get_yhat(model,dl_,device=None,f_transform_x=None):
    with torch.no_grad():
        i_ae = []
        t=0
        for step, (xb, yb) in enumerate(dl_):
            if device is not None:
                xb = xb.to(device)
                if f_transform_x is not None:
                    xb = f_transform_x(xb)
                yhat_b = model(xb).detach().cpu()
            else:
                if f_transform_x is not None:
                    xb = f_transform_x(xb)
                yhat_b = model(xb).detach().numpy()
            if t == 0:
                yhat = yhat_b
                y = yb.detach().numpy()
            else:
                yhat = np.append(yhat, yhat_b, axis=0)
                y    = np.append(y,yb.detach().numpy())
            t+=1
    return yhat.squeeze(), y.squeeze()

    
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


def f_get_p_model(model):
    p_model = 0
    for p in model.parameters():
        if len(p.shape)>1:        # matrix          : array 2 dims
            p_model += p.shape[0] * p.shape[1]
        else:
            p_model += p.shape[0] # vector or scalar:  array 1 dim
    return p_model
    

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

def f_fit_poisson(X_,y_,n_epoqs=800, size_batch=50, alpha_t = 1e-4, 
                  diff=1e-5, alpha=0.01, show=1, algo="nr",bound=2):
    start_time = time.time()
    
    n_, p_     = X_.shape
    y_         = y_.reshape(n_,1)
    beta_init  = 0*(np.random.uniform(size=p_,low=0.01,high=1.0)/p_-0.5/p_).reshape(p_,1)
    beta       = beta_init.reshape(p_,1)
    mu         = np.exp(X_ @ beta)
    logL       = np.sum(y_ * np.log(mu) - mu - gammaln(y_+1))
    logLik_s   = []
    logLik_s.append(logL)
    
    ## with minibatches..
    if algo!="nr" and algo!="ef":
        H_old  = alpha * np.eye(len(beta))
        i_s = np.array(np.mgrid[0:n_:complex(real=0,imag=size_batch)],dtype=int)   
        if algo=="efseq1" or algo=="efseq1_diag": \
            Hinv=Hinv_old= (1/alpha) * np.eye(len(beta))
    ## with full batch
    if algo=="nr" or algo=="ef":
        H_old      = 0.001 * np.eye(len(beta))
        #ones       = np.ones((n_,1)).reshape((n_,1))
    ##
    if algo=="efseq":
        updater_mnng = UpdaterMinibatchNaturalGradient(p_,alpha)
    ##
                
    for epoch in range(0,n_epoqs):
        
        ##-------  BEGIN FULL-BATCHES algoS ----------------------------------------
        if algo=="nr" or  algo=="nr_diag": 
            f_hessian_poisson = f_hessian_poisson_newton        
        
        if algo=="ef" or algo=="ef_diag": 
            f_hessian_poisson = f_hessian_poisson_natural
        
        if algo=="nr" or algo=="ef" \
            or algo=="nr_diag" or algo=="ef_diag":
            mu       = np.exp(X_ @ beta)
            g        = f_gradient_poisson(X_,y_,mu)       #### X_.T @ (y_ - mu)    # gradient
            H        = f_hessian_poisson(X_,y_,mu,g,H_old) #### -(X_.T @ (mu * X_)) # hessian
            H        = H - alpha*np.eye(len(H))
            
            if algo=="nr_diag" or algo=="ef_diag":
                Hinv = np.zeros((p_,p_))
                for k in range(p_): Hinv[k,k] = 1/H[k,k]
                beta_new = beta - alpha_t*(Hinv @ g) #only diagonal hessian
            else:
                Hinv = np.linalg.inv(H)
                beta_new = beta - (Hinv @ g) #eventually add alpha_t too here
        ##-------  END FULL-BATCHES algoS -------------------------------------------
        
        ##-------  BEGIN MINI-BATCHES algoS ----------------------------------------
        if algo=="gd" or algo=="efseq1"\
                or algo=="efseq1_diag" or algo=="efseq": 
            if algo=="efseq1" or algo=="efseq1_diag":
                 #or algo=="efseq"  : 
                Hinv=Hinv_old= (1/alpha) * np.eye(len(beta))
            #Loop minibatch (no shuffling here - to be added!)
            for l in range(0,len(i_s)-1):
                sb           = range(i_s[l]+1*(l>0),i_s[l+1])
                Xb           = X_[sb,:]
                yb           = y_[sb]
                mub          = np.exp(Xb @ beta)
                #grad_C_b     = -Xb.T @ (yb - mub)
                if algo=="gd":
                    gradient_b = -Xb.T @ (yb - mub)
                    gradient_b[gradient_b>bound]  = bound
                    gradient_b[gradient_b<-bound] = -bound
                    beta_new   = beta - alpha_t * gradient_b #- 0.015 * beta
#                 if algo=="newton_mb":
#                     mu           = np.exp(X_ @ beta)
#                     #H            = - f_hessian_poisson(X_,y_,mu,grad_C_b,H_old) ####
#                     H            = (X_.T @ (mu * X_))
#                     H            = H - alpha*np.eye(len(H))  
#                     gradient_b   = -Xb.T @ (yb - mub)
#                     gradient_b[gradient_b>5] = 5
#                     gradient_b[gradient_b<-5] = -5
#                     beta_new     = beta - alpha_t *(np.linalg.inv(H) @ gradient_b)
                if algo=="efseq1" or algo=="efseq1_diag":
                    gradient_b   = Xb.T @ np.diag(v=(yb-mub).ravel())
                    gradient_b[gradient_b>bound]  = bound
                    gradient_b[gradient_b<-bound] = -bound
                    for i in range(gradient_b.shape[1]):
                        gb_i   = np.sqrt(mub[i])*gradient_b[:,i].reshape((p_,1))  #- 0.015 * beta
                        if l>0: a = i_s[l]/i_s[l+1]
                        if l==0: a=1
                        b = 1/i_s[l+1]
                        Hinv   = a*Hinv_old - \
                          b/(1+b*gb_i.T @ Hinv @gb_i) * Hinv @ gb_i @ gb_i.T @ Hinv                        
                    ##
                    if algo=="efseq1_diag":
                        for k1 in range(p_):
                            for k2 in range(p_):
                                if k1!=k2:
                                    Hinv[k1,k2] = 0.0
                  ##
                  ##
                    gradient_b = - np.sum(gradient_b,axis=1).reshape((p_,1)) #-Xb.T @ (yb - mub)
                    beta_new   = beta - alpha_t *Hinv @ gradient_b #- 0.0015 * beta
#                if algo=="efseq12":
#                    pass #keep same approximated hessian from previous epoch for update
                         #while use gradients vectors to compute next approximated hessian
#                 if algo=="efseq1_diag":
#                     gradient_b   = Xb.T @ np.diag(v=(yb-mub).ravel())
#                     gradient_b[gradient_b>bound]  = bound
#                     gradient_b[gradient_b<-bound] = -bound
#                     for i in range(gradient_b.shape[1]):
                ## --------------------------------------    
                if algo=="efseq":
#                     gradient_b   = Xb.T @ np.diag(v=(yb-mub).ravel())
#                     gradient_b[gradient_b>bound]  = bound
#                     gradient_b[gradient_b<-bound] = -bound
#                     Gl = gradient_b
#                     Il = np.eye(Gl.shape[1])                    
#                     if l>1:
#                         al = i_s[l]/i_s[l-1]
#                         bl = 1/i_s[l]
#                     if l<=1: 
#                         al = 1
#                         bl = 1
#                     IGHGinv    = np.linalg.inv(Il+bl*Gl.T@Hinv@Gl)
#                     Hinv       = al*(Hinv-bl*Hinv@Gl@IGHGinv@Gl.T@Hinv)
#                     gradient_b = - np.sum(gradient_b,axis=1).reshape((p_,1))
#                     update_b   = Hinv @ gradient_b
                    if l==0: updater_mnng.resetHinv()
                    gradient_b  = Xb.T @ np.diag(v=(yb-mub).ravel())
                    gradient_b[gradient_b>bound]  = bound
                    gradient_b[gradient_b<-bound] = -bound
                    update_b = updater_mnng.update(l,i_s,gradient_b)
                    beta_new   = beta - alpha_t * update_b
                ##
                ##
        ##-------  END MINI-BATCHES algoS --------------------------------------------
        ##
        normbb   = np.sqrt(np.sum((beta_new - beta)**2)/len(beta))
        #print("normbb=",normbb)
        beta     = beta_new
        mu       = np.exp(X_ @ beta)
        logL     = np.sum(y_ * np.log(mu) - mu - np.log(factorial(y_)))
        logLik_s.append(logL)
        ##
        if epoch>2 and logLik_s[-1]<logLik_s[-2]:
            #if algo!="efseq":
            alpha_t = alpha_t/2
            #else:
            #    alpha_t = alpha_t/3
        ##
        if show>1:
            print(f'Iter n°: {epoch:2d}', f'Log_lik = {logLik_s[-1]:2.4f}', \
                  f'Dist_beta = {normbb:2.5f}',f'beta_hat[0] = {np.round(beta.flatten()[0],3)}')
        ##
        if normbb<diff and epoch>10:
            break
        else:
            if algo=="efseq1" or algo=="efseq1_diag": Hinv_old=Hinv
            if algo=="nr" or algo=="ef": H_old=H
    ##
    step_end = epoch
    beta_hat = beta
    logL_hat = logL
    ##
    mu_hat = np.exp(X_ @ beta_hat)
    #g_hat = X.T @ (y - mu_hat)
    H_hat = -(X_.T @ (mu_hat * X_))
    #
    std_hat = np.round(np.sqrt(np.diag(np.linalg.inv(-H_hat))),4)
    if show>=1:
        #print( 'std_hat    =',std_hat)
        print(f'beta_hat = {np.round(beta_hat.flatten(),3)} logL_hat={np.round(logL_hat,3)}') 
    ##
    return {"beta":beta_hat, "std":std_hat, 
            "logL":logL_hat, "mu":mu_hat, 
            "step_end":step_end, "logL_s":logLik_s,
            "algo":algo, "time":(time.time() - start_time)}


def f_poisson_logLik(beta,X,y,name=None):
    beta = beta.reshape(len(beta),1)
    y = y.reshape((len(y),1)).astype(np.float64)
    mu_hat  = np.exp(X @ beta) #.ravel()
    logL    = np.sum(y * np.log(mu_hat) - mu_hat - gammaln(y+1))
    if name is not None: print(name+"=",np.round(logL,4))
    return logL

def f_mu_mse_cor_poisson(X,y,fit,mu0,isprint=None):
    beta = fit["beta"]
    algo = fit["algo"]
    mu_hat  = np.exp(X @ beta).ravel()
    mse_mu_hat = ( (mu_hat-mu0.ravel())**2 ).mean()
    cor_mu_hat = np.corrcoef(mu0.ravel(),mu_hat)[0,1]
    logLik = f_poisson_logLik(beta,X,y)
    
    if isprint is not None:
        print(str("mse_mu_"+algo+"="),np.round(mse_mu_hat,4), 
              #str("cor(cor_mu_"+namethod+",mu)="),np.round(cor_mu_hat,4),
              str("logLik_"+algo+"="),np.round(logLik,4))
    
    return {"mu":mu_hat, "msemu":mse_mu_hat, 
            "cormu":cor_mu_hat, "logL":logLik,
            "fit":fit}





def f_save_poisson_sample_Xy_in_memmap(filename_n_d_, beta, n = 5000, p=10):
    p1=len(beta)
    p=p1-1
    
    size_minibatch = 200

    X_map = np.memmap(str(filename_n_d_+"X.memmap"), \
                        dtype='float32', mode='w+', shape=(n,p1))
    
    y_map = np.memmap(str(filename_n_d_+"y.memmap"), \
                        dtype='float32', mode='w+', shape=(n,1))
    
                      
    X_map[:,0] = np.ones(n).ravel()


    for idx_b in range(0, n, size_minibatch):
        idx_b2 = np.min( [idx_b+size_minibatch,n] )
        nb=idx_b2-idx_b
        Xb=np.random.uniform(size=nb*p,
            low=0,high=1).reshape(nb,p)/p1/5
        Xb = np.hstack([np.ones((Xb.shape[0],1)),Xb])
        X_map[idx_b:idx_b2,0:p1] = Xb

        mub = np.exp(Xb@beta)
        yb  = np.random.poisson(lam=mub)
        y_map[idx_b:idx_b2,0] = yb.ravel()

    # add correlations?
    # import random as rd
    # for l in rand(10):
    #     j1 = rd.randint(1, p1)
    #     nj2 = rd.randint(1, 7)
    #     j2s = rd.sample(range(1, p1), nj2)
    #     j1, j2s
    #     noise = np.random.uniform(size=n,low=0,high=1).reshape((n,1))/len(beta)/10
    #     X_map[:,j1] = np.mean(X_map[:,j2s],axis=1) + noise.ravel()

    del X_map, y_map
    

# save dataloader into memmap file (numpy)
# save into memmap format from numpy a dataloader (2 arrays)
# x,y from dataloader et nb size of a minibatch
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
    
    
def f_read_memmap(filename_x,n,p):
    x_map = np.memmap(filename_x, dtype='float32', 
                      mode='r', shape=(n,p))
    return x_map
    
    
    
def f_barplot_poisson(y,printed_bars=True,printed_counts=True,labely="y"):
    from collections import Counter as counter
    dico_y = counter(y.squeeze().tolist())    
    keys_y = np.sort([ int(k) for k in dico_y.keys()])
    
    if printed_counts:
        print([ (int(k),dico_y[k]) for k in keys_y])

    if printed_bars:
        import matplotlib.pyplot as plt
        import seaborn; seaborn.set() # set plot style
        plt.bar(dico_y.keys(),dico_y.values(),color="blue")
        plt.title(r'Discrete distribution for '+labely)
        plt.xlabel(r' ') #plt.xlabel(r'$(rings$')
        plt.ylabel(r'Counts')
        plt.hlines(0,1,2)
        plt.show()
    
    return dico_y, keys_y


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



          
def f_save_dataloader_to_h5py(dataloader,filename,d,transformx=None):  

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
                              show=0,batch_size=100):  
    
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


def f_ipca_from_h5py(filename,n_components=2,batch_size=10,chunk_size=100):  

    file = h5py.File( filename , 'r' )
    data_y= file[ 'y' ]
    data_x= file[ 'x' ] 
    n    = data_x.shape[ 0 ]  #; print(n)
    d    = data_x.shape[ 1 ]  #; print(d)
    ipca = IncrementalPCA(n_components= n_components , batch_size= batch_size )
    
    for i in range ( 0 , n//chunk_size):
        ipca.partial_fit(data_x[i*chunk_size : (i+ 1 )*chunk_size,:])
        y_i = data_y[i*chunk_size : (i+ 1 )*chunk_size]
        if i==0:
            y = y_i.reshape((len(y_i),1))
        else:
            y = np.append(y,y_i, axis=0)
    
    if len(y)<n:
        next_i = (n//chunk_size)*chunk_size
        y_i = data_y[next_i:n]
        y = np.append(y,y_i, axis=0)
    
    file.close()
    
    return ipca, y, n, d

    
def f_projection_from_ipca_from_h5py(filename,ipca,z_ipca,chunk_size):

    file = h5py.File( filename , 'r' )
    data_x = file[ 'x' ] 
    n    = data_x.shape[ 0 ]  #; print(n)
    d    = data_x.shape[ 1 ]  #; print(d)
    data_y = file[ 'y' ] 
    
    y = np.zeros((n,1))
    
    for i in range ( 0 , n//chunk_size):
        z_ipca_i = ipca.transform(data_x[i*chunk_size : (i+ 1 )*chunk_size,:])
        y_i = data_y[i*chunk_size : (i+ 1 )*chunk_size]
#         if i == 0:
#             y = y_i.reshape((len(y_i),1))
#             #z_ipca = z_ipca_i
#             #z_ipca[i*chunk_size : (i+ 1 )*chunk_size,:] = z_ipca_i
#             #print(data_x[i*chunk_size : (i+ 1 )*chunk_size,:].shape)
#             #print(z_ipca_i.shape)
#         else:
#             y = np.append(y,y_i, axis=0)
#             #z_ipca = np.append(z_ipca,z_ipca_i, axis=0)
        y[i*chunk_size : (i+ 1 )*chunk_size] = y_i.reshape((len(y_i),1))    
        z_ipca[i*chunk_size : (i+ 1 )*chunk_size,:] = z_ipca_i
    
    #if z_ipca.shape[0]<n:
    if len(y)<n:
        next_i = (n//chunk_size)*chunk_size
        y_i = data_y[next_i:n]
        #y = np.append(y,y_i, axis=0)
        y[next_i:n,:] = y_i.reshape((len(y_i),1))  
        z_ipca_i = ipca.transform(data_x[next_i:n,:])
        #z_ipca = np.append(z_ipca,z_ipca_i, axis=0)
        z_ipca[next_i:n,:] = z_ipca_i
    
    return z_ipca, y


def f_score_projection(z,y,name="",show=False):
    y = y.ravel()
    db = davies_bouldin_score(z, y)
    sl = silhouette_score(z, y)
    if show is True:
        print("Davies_Bouldin_score of",name, "=",np.round(db,3), 
              "\nSilhouette_score     of",name, "=",np.round(sl,3))
    return db, sl


def f_projection_from_openTSNE(x_map,x_map_init,perplexity=30,n_jobs=1,
                               random_state=0,verbose=False, draw=True,
                               title_draw = "openTSNE ouput", quality=True, 
                               text_quality=None, i_s_quality=None):
    
    aff = openTSNE.affinity.PerplexityBasedNN(
        x_map,
        perplexity=perplexity,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    z_init_openTSNE = openTSNE.initialization.rescale(x_map_init)
    
    #%%time
    z_tsne = openTSNE.TSNE(
        n_jobs=n_jobs,
        verbose=verbose,
    ).fit(affinities=aff, initialization=z_init_openTSNE)
    
    return z_tsne, z_init_openTSNE, aff
    

class AutoEncoder(nn.Module):
    def __init__(self, name, layers_encoder, layers_decoder, init_layers = None):
        super().__init__()
        self.name = name
        self.layers_encoder = layers_encoder
        self.layers_decoder = layers_decoder
        self.net_encoder = nn.Sequential(*layers_encoder)
        self.net_decoder = nn.Sequential(*layers_decoder)
        self.init_layers = init_layers
        if self.init_layers is not None:
            for k in self.init_layers:
                torch.nn.init.xavier_uniform_(self.net_encoder[k].weight)
                torch.nn.init.xavier_uniform_(self.net_decoder[k].weight)        
    
    def forward(self, x):
        encoded = self.net_encoder(x)
        decoded = self.net_decoder(encoded)
        return decoded
    
    def encoder(self,x):
        encoded = self.net_encoder(x)
        return encoded
    
    def decoder(self,z):
        decoded = self.net_decoder(z)
        return decoded


def f_train_autoencoder(dl_train,autoencoder,nbmax_epoqs,lr,device=None,epoch_print=5):
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss = nn.MSELoss(reduction='sum')
    loss_s = np.zeros(nbmax_epoqs)
    if device is not None: autoencoder=autoencoder.to(device)
    autoencoder.train()
    t=0
    for epoch in range(nbmax_epoqs):
        loss_t = 0 
        for step, tuple_b in enumerate(dl_train):
            xb = tuple_b[0]
            yb = tuple_b[1]
            if device is not None:
                xb=xb.to(device)
                yb=yb.to(device)
            xb_hat = autoencoder(xb)
            lossb = loss(xb_hat, xb)       
            optimizer.zero_grad()               
            lossb.backward()                     
            optimizer.step()
            loss_t += lossb
        loss_s[t] = loss_t
        if epoch % epoch_print == 0 or (epoch == nbmax_epoqs-1 and epoch_print<=nbmax_epoqs):
            print("t=",t,"\tloss=",np.round(loss_t.detach().cpu().numpy(),5))
        t+=1
    
    autoencoder.eval()
    tmax = t
    return loss_s, tmax


def f_train_fromh5_autoencoder(dl_train,autoencoder,nbmax_epoqs,
                               lr,  epoch_print=3,  device=None,
                               transform_x=None,loss_yy_model=None):
    autoencoder = autoencoder.to(device)
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    #optimizer = torch.optim.SGD(autoencoder.parameters(), lr=lr)
    loss_func = nn.MSELoss(reduction='sum')
    # loss_func = nn.MSELoss()
    loss_s = np.zeros(nbmax_epoqs)
    for epoch in range(nbmax_epoqs):
        loss_b = 0 
        for step, tuple_b in enumerate(dl_train):
            ib = None
            xb = tuple_b[0]
            yb = tuple_b[1]            
            if len(tuple_b)==3: ib = tuple_b[2]
            #
            if device is not None: xb = xb.to(device)
            if transform_x is not None: xb=transform_x(xb) 
            xb_hat = autoencoder(xb)
            loss = loss_func(xb_hat, xb)
            if loss_yy_model is not None:
                loss = loss_yy_model(loss,ib,xb,yb,xb_hat,autoencoder)
            optimizer.zero_grad()               
            loss.backward()                     
            optimizer.step()
            loss_b += loss
        loss_s[epoch] = loss_b.detach().cpu().numpy()
        if epoch%epoch_print==0 or (epoch==nbmax_epoqs-1 and epoch_print<=nbmax_epoqs):
            print("epoch=",epoch,"\tloss=",np.round(loss_s[epoch],5))
        autoencoder.eval()
    return loss_s, epoch+1


def f_get_latent_space_autoencoder(autoencoder,dl_train,
                                   device=None,transform_x=None,):
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        i_ae = []
        t=0
        for step, tuple_b in enumerate(dl_train): #xb,yb,ib
            ib = None
            xb = tuple_b[0]
            yb = tuple_b[1]
            if len(tuple_b)==3: ib = tuple_b[2]
            #
            if device is not None: xb = xb.to(device)            
            if transform_x is not None: xb=transform_x(xb) 
            zb_hat = autoencoder.encoder(xb).detach().cpu().numpy()
            if t == 0:
                z_ae = zb_hat
                y_ae = yb.detach().numpy()
                if ib is not None: i_ae = ib
            else:
                z_ae = np.append(z_ae, zb_hat, axis=0)
                y_ae = np.append(y_ae,yb.detach().numpy())
                if ib is not None: i_ae = np.append(i_ae,ib)
            t+=1
    return z_ae, y_ae, i_ae


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

def f_save_to_reduction_to_memmap_files(filename_xin,
                                        filename_xout,
                                        R,n= None,
                                        size_minibatch = 50):
    
    p_in, p_out = R.shape
    
    xin_map  = np.memmap(filename_xin, dtype='float32', 
                         mode='r', shape=(n,p_in))
    
    xout_map = np.memmap(filename_xout, dtype='float32', 
                         mode='w+', shape=(n,p_out))
    
    for idx_b in range(0, n, size_minibatch):
        idx_b2 = np.min( [idx_b+size_minibatch,n] )
        xb = xin_map[idx_b:idx_b2,:]        
        xout_map[idx_b:idx_b2,:] = xb @ R
    
    np.savetxt(str(filename_xin+".shape.txt"),[n,p_in])
    np.savetxt(str(filename_xout+".shape.txt"),[n,p_out])
    
    del xin_map, xout_map
    
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


