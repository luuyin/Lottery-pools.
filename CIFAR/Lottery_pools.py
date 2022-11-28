'''
    main process for retrain a subnetwork from beginning
'''
import os
import pdb
import time 
import pickle
import random
import shutil
import argparse
import numpy as np  
from copy import deepcopy
import matplotlib.pyplot as plt
import copy
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import re
import os
import math
from utils import *
from pruner import *
import numpy as np
import math
 

parser = argparse.ArgumentParser(description='PyTorch Training Subnetworks')

##################################### Dataset #################################################
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--input_size', type=int, default=32, help='size of input images')

##################################### Architecture ############################################
parser.add_argument('--arch', type=str, default='resnet20s', help='model architecture')
parser.add_argument('--imagenet_arch', action="store_true", help="architecture for imagenet size samples")

##################################### General setting ############################################
parser.add_argument('--seed', default=None, type=int, help='random seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataloader')
parser.add_argument('--resume', action="store_true", help="resume from checkpoint")
parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint file')
parser.add_argument('--inference', action="store_true", help="testing")
parser.add_argument('--save_dir', help='The directory used to save the trained models', default=None, type=str)

##################################### Training setting #################################################
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=182, type=int, help='number of total epochs to run')
parser.add_argument('--warmup', default=0, type=int, help='warm up epochs')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='91,136', help='decreasing strategy')

##################################### Pruning setting #################################################
parser.add_argument('--tickets_mask', default=None, type=str, help='mask for subnetworks')
parser.add_argument('--tickets_init', default=None, type=str, help='initilization for subnetworks')



##################################### Pruning setting #################################################
parser.add_argument('--pruning_times', default=19, type=int, help='overall times of pruning')
parser.add_argument('--rate', default=0.2, type=float, help='pruning rate')
parser.add_argument('--prune_type', default='lt', type=str, help='IMP type (lt, pt or rewind_lt)')
parser.add_argument('--random_prune', action='store_true', help='whether using random prune')
parser.add_argument('--rewind_epoch', default=3, type=int, help='rewind checkpoint')





##################################### Lottery Pools #################################################
parser.add_argument('--search_num', default=3, type=int, help=' the count of candidate lotter pools for interpolation')
parser.add_argument('--EMA_value', default=0.9, type=float, help='EMA factor for interpolation')
parser.add_argument('--interpolate_method',  type=str , default="Lottery_pools", choices=['Lottery_pools','interpolate_ema', 'interpolate_swa'], help='interpolate_LTs method')
parser.add_argument('--interpolation_value_list', type=float, nargs='+',help="the  candidate coefficient pools for interpolation")




best_sa = 0



def model_files_filter(model_files,filter_itrs=["best"]):
    new_files=[]
    for filter_itr in filter_itrs:
        for  model_file in model_files:
            if filter_itr in model_file:
                new_files.append(model_file)
    return new_files



def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)







def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (image, target) in enumerate(val_loader):
        
        image = image.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(image)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
        .format(top1=top1))

    return top1.avg,losses.avg

def save_checkpoint(state, is_SA_best, save_path, filename='checkpoint.pth.tar'):
    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)
    if is_SA_best:
        shutil.copyfile(filepath, os.path.join(save_path, 'model_SA_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    print('setup random seed = {}'.format(seed))
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 



import copy 
def extract_mask(model_dict):

    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_key=key[:-5]
            new_dict[new_key] = copy.deepcopy(model_dict[key])

    return new_dict


def extract_weight(model_dict):

    new_dict = {}
    for key in model_dict.keys():
        if 'mask' not in key:
            if "orig" in key:
                new_key=key[:-5]
            else:
                new_key=key
            new_dict[new_key] = copy.deepcopy(model_dict[key])

    return new_dict


def apply_mask(model,masks):

    for name, tensor in model.named_parameters():
        if name in masks:
            tensor.data = tensor.data*masks[name]






### weight avarege


def get_model_params(model):
    params = {}
    for name in model.state_dict():
        params[name] = copy.deepcopy(model.state_dict()[name])
    return params

def set_model_params(model, model_parameters):
    model.load_state_dict(model_parameters)


def interpolate_LTs(interpolation):

    def function(LTs_solutions,rangelist,interpolation=interpolation):
        print ("interpolation",interpolation)
        interpolation_flag="moving_"+str(interpolation)
        params = {}
        pre_params = {}
        for name in LTs_solutions[0].state_dict():
            pre_params[name] = copy.deepcopy(LTs_solutions[0].state_dict()[name])

        rangelist=list(rangelist)[1:]
        for name in LTs_solutions[0].state_dict():
            for i in rangelist:
                params[name]=copy.deepcopy(LTs_solutions[i].state_dict()[name]* interpolation+pre_params[name]* (1 - interpolation))
                pre_params[name]=params[name]                    
                                            
        return params,interpolation_flag
    return function




def main():
    global args, best_sa
    args = parser.parse_args()
    print(args)



    torch.cuda.set_device(int(args.gpu))
    if not args.inference:
        os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset 
    model, train_loader, val_loader, test_loader = setup_model_dataset(args)

    if args.tickets_init:
        print('loading init from {}'.format(args.tickets_init))
        init_file = torch.load(args.tickets_init, map_location='cpu')
        if 'init_weight' in init_file:
            init_file = init_file['init_weight']
        model.load_state_dict(init_file)

    # setup initialization and mask 
    if args.tickets_mask:
        print('loading mask from {}'.format(args.tickets_mask))
        mask_file = torch.load(args.tickets_mask, map_location='cpu')
        if 'state_dict' in mask_file:
            mask_file = mask_file['state_dict']
        mask_file = extract_mask(mask_file)
        print('pruning with {} masks'.format(len(mask_file)))
        prune_model_custom(model, mask_file)

    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)



    if args.inference:
        # test

        model_files = os.listdir(args.checkpoint)
        model_files=model_files_filter(model_files)
        model_files = sorted_nicely(model_files)
    #     model_files=list(reversed(model_files))
        print ("model_files",model_files)

        sparsity_list=[]
        LTs_solutions=[]
        ori_acc=[]
    #     model_files=[model_files[3],model_files[5],model_files[7]]
        model_files=model_files
        for model_file in model_files:
            print ("model_file",model_file)
            ## init model
            
            model, train_loader, val_loader, test_loader = setup_model_dataset(args)
            model=model.cuda()


            checkpoint = torch.load(os.path.join(args.checkpoint,model_file), map_location = torch.device('cuda:'+str(args.gpu)))
            if 'state_dict' in checkpoint.keys():
                checkpoint = checkpoint['state_dict']

            current_mask =extract_mask(checkpoint)
            current_weight =extract_weight(checkpoint)
            model.load_state_dict(current_weight)
            apply_mask(model,current_mask)

            print ("===============")

            test_acc,test_loss = validate(test_loader, model, criterion) 
            sparsity=check_sparsity(model)
            print('* Test Accurayc = {}'.format(test_acc))
            
            
            LTs_solutions.append(model_file)
            sparsity_list.append(sparsity)
            ori_acc.append(test_acc)


    print ("inference done")
    print ("\n")





    def load_check_point(model_file):

        loaded_model = setup_model(args)
        loaded_model=loaded_model.cuda()


        checkpoint = torch.load(os.path.join(args.checkpoint,model_file), map_location = torch.device('cuda:'+str(args.gpu)))
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        current_mask =extract_mask(checkpoint)
        current_weight =extract_weight(checkpoint)
        loaded_model.load_state_dict(current_weight)
        apply_mask(loaded_model,current_mask)
        
        return loaded_model


    def liner_inter(use_LTs_solutions,new_indx):

        new_indx=np.array(new_indx)

        print ("begin liner_inter")
        print ("prune_type",args.prune_type)

        
        
        all_acc=[]
        all_loss=[]
        
        for i in range(len(use_LTs_solutions)):  



            model_pre = load_check_point (use_LTs_solutions[i])
            model_current= load_check_point( use_LTs_solutions[i+1])

            current_tickets_model=[model_pre,model_current]


            print ("================")

            print ("inter inex",i,"to",i+1)



            inter_num=10
            inter_value_list=[i/inter_num for i in range(inter_num) ]

            print ("beging search interpolate value")
            for i in range(len(inter_value_list)):
                method=interpolate_LTs(inter_value_list[i])

                interpolation_model = setup_model(args)
                interpolation_model=interpolation_model.cuda()
                params,interpolation_flag = method(current_tickets_model,range(len(current_tickets_model)))
                set_model_params(interpolation_model, params)

        #        "update_bn"

                torch.optim.swa_utils.update_bn(train_loader, interpolation_model,"cuda")
        #         print ("update_bn done")
                test_moving_acc,test_moving_loss = validate(test_loader, interpolation_model, criterion) 
                print('* Test Accurayc = {}'.format(test_moving_acc))
                print('* Test loss = {}'.format(test_moving_loss))

                all_acc.append(test_moving_acc)
                all_loss.append(test_moving_loss)
                
            check_sparsity(interpolation_model)

        return all_acc,all_loss






    def interpolate_ema(use_LTs_solutions,start,new_indx):
        
        new_indx=np.array(new_indx)
        if start==0: density=None 
        else: density=(sparsity_list[start])/100
        lth_acc=ori_acc[start]


        print ("\n")
        print ("creating interpolation model")


        print ("number of tickets",len(use_LTs_solutions))



        print ("begin interpolation using ema")

        best_acc=lth_acc



        current_ticket=use_LTs_solutions[0]
        
        current_ticket=load_check_point(current_ticket)

        to_pend_ind=[0]
        best_interpolation_all=[]

    #     for i in range(1,len(use_LTs_solutions)): 


        for i in range(1,min(len(use_LTs_solutions),args.search_num)  ):  
            
            
            new_ticket=use_LTs_solutions[i]
            
            new_ticket=load_check_point(new_ticket)
            
            current_tickets_model=[ current_ticket,new_ticket]
            
            current_ticket_acc,_ = validate(test_loader, current_ticket, criterion) 
            new_ticket_acc,_ = validate(test_loader, new_ticket, criterion) 
            
            
            current_ticket_ind=copy.deepcopy(to_pend_ind)
            current_ticket_ind.append(i)
                
            

            print ("================")

            print ("search inex",i,"current search tickets len",len(current_ticket_ind),new_indx[np.array(current_ticket_ind)])

            
            
            
            
            moving_value=[args.EMA_value]
            # init model 
            best_moving_acc=0
            best_interpolation=0

            
            print ("beging search interpolate value")
            for i in range(len(moving_value)):
                method=interpolate_LTs(moving_value[i])
                
                interpolation_model = setup_model(args)
                interpolation_model=interpolation_model.cuda()
                params,interpolation_flag = method(current_tickets_model,range(len(current_tickets_model)))
                set_model_params(interpolation_model, params)



                ### prune
            
                if density!=None:
    
                    weight_abs = []

                    for name, weight in interpolation_model.named_parameters():
                        if name not in current_mask: continue
                        weight_abs.append(torch.abs(weight))

                    # Gather all scores in a single vector and normalise
                    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
                    num_params_to_keep = int(len(all_scores) * density)

                    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
                    acceptable_score = threshold[-1]


                    for name, weight in interpolation_model.named_parameters():
                        if name not in current_mask: continue
                        current_mask[name][:] = ((torch.abs(weight)) >= acceptable_score).float()


                    for name, tensor in interpolation_model.named_parameters():
                        if name in current_mask:
                            tensor.data = tensor.data * current_mask[name]



        #        "update_bn"

                torch.optim.swa_utils.update_bn(train_loader, interpolation_model,"cuda")
        #         print ("update_bn done")
                test_moving_acc,_ = validate(test_loader, interpolation_model, criterion) 
                print('* Test Accurayc = {}'.format(test_moving_acc))
                
                if test_moving_acc > best_moving_acc:
                    best_interpolation_model=copy.deepcopy(interpolation_model)
                    
                    best_moving_acc=test_moving_acc
                    best_interpolation=moving_value[i]
                
                    print ("interpolation ratio",best_interpolation,"acc",best_moving_acc)
                
                
                print ("\n")
                


            best_acc=best_moving_acc
            print ("best_acc",best_moving_acc)

            current_ticket=copy.deepcopy(best_interpolation_model)
            
            to_pend_ind=current_ticket_ind
            best_interpolation_all.append(best_interpolation)
        
            check_sparsity(best_interpolation_model)
            


            
            
            
        print('*** LTH Accurayc = {}'.format(lth_acc))
        print ("ema acc",best_acc)

        print ("**************")




    def interpolate_swa(use_LTs_solutions,start,new_indx):
        
        new_indx=np.array(new_indx)
        if start==0: density=None 
        else: density=(sparsity_list[start])/100
        lth_acc=ori_acc[start]


        print ("\n")
        print ("creating interpolation model")


        print ("number of tickets",len(use_LTs_solutions))



        print ("begin interpolation using swa")

        best_acc=lth_acc



        current_ticket=use_LTs_solutions[0]
        current_ticket=load_check_point(current_ticket)        
        
        to_pend_ind=[0]
        best_interpolation_all=[]

    #     for i in range(1,len(use_LTs_solutions)): 


        for i in range(1,min(len(use_LTs_solutions),args.search_num)  ):  
            
            
            new_ticket=use_LTs_solutions[i]
            new_ticket=load_check_point(new_ticket)
            
            
            current_tickets_model=[ current_ticket,new_ticket]
            
            current_ticket_acc,_ = validate(test_loader, current_ticket, criterion) 
            new_ticket_acc,_ = validate(test_loader, new_ticket, criterion) 
            
            
            current_ticket_ind=copy.deepcopy(to_pend_ind)
            current_ticket_ind.append(i)
                
            

            print ("================")

            print ("search inex",i,"current search tickets len",len(current_ticket_ind),new_indx[np.array(current_ticket_ind)])

            
            
            
            
            moving_value=[1/(len(current_ticket_ind))]
            # init model 
            best_moving_acc=0
            best_interpolation=0

            
            print ("beging search interpolate value")
            for i in range(len(moving_value)):
                method=interpolate_LTs(moving_value[i])
                
                interpolation_model = setup_model(args)
                interpolation_model=interpolation_model.cuda()
                params,interpolation_flag = method(current_tickets_model,range(len(current_tickets_model)))
                set_model_params(interpolation_model, params)



                ### prune
            
                if density!=None:
    
                    weight_abs = []

                    for name, weight in interpolation_model.named_parameters():
                        if name not in current_mask: continue
                        weight_abs.append(torch.abs(weight))

                    # Gather all scores in a single vector and normalise
                    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
                    num_params_to_keep = int(len(all_scores) * density)

                    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
                    acceptable_score = threshold[-1]


                    for name, weight in interpolation_model.named_parameters():
                        if name not in current_mask: continue
                        current_mask[name][:] = ((torch.abs(weight)) >= acceptable_score).float()


                    for name, tensor in interpolation_model.named_parameters():
                        if name in current_mask:
                            tensor.data = tensor.data * current_mask[name]



        #        "update_bn"

                torch.optim.swa_utils.update_bn(train_loader, interpolation_model,"cuda")
        #         print ("update_bn done")
                test_moving_acc,_ = validate(test_loader, interpolation_model, criterion) 
                print('* Test Accurayc = {}'.format(test_moving_acc))
                
                if test_moving_acc > best_moving_acc:
                    best_interpolation_model=copy.deepcopy(interpolation_model)
                    
                    best_moving_acc=test_moving_acc
                    best_interpolation=moving_value[i]
                
                    print ("interpolation ratio",best_interpolation,"acc",best_moving_acc)
                
                
                print ("\n")
                


            best_acc=best_moving_acc
            print ("best_acc",best_moving_acc)

            current_ticket=copy.deepcopy(best_interpolation_model)
            
            to_pend_ind=current_ticket_ind
            best_interpolation_all.append(best_interpolation)
        
            check_sparsity(best_interpolation_model)
            


            
            
            
        print('*** LTH Accurayc = {}'.format(lth_acc))
        print ("swa acc",best_acc)

        print ("**************")



    def Lottery_pools(use_LTs_solutions,start,new_indx):
        
        new_indx=np.array(new_indx)
        if start==0: density=None 
        else: density=(sparsity_list[start])/100
        lth_acc=ori_acc[start]


        print ("\n")
        print ("creating interpolation model")


        print ("number of tickets",len(use_LTs_solutions))



        print ("begin greedy search")

        best_acc=lth_acc



        current_ticket=use_LTs_solutions[0]
        current_ticket=load_check_point(current_ticket)    
        
        to_pend_ind=[0]
        best_interpolation_all=[]

    #     for i in range(1,len(use_LTs_solutions)): 


        for i in range(1,min(len(use_LTs_solutions),args.search_num)  ):  
            
            
            new_ticket=use_LTs_solutions[i]
            new_ticket=load_check_point(new_ticket)    
            
            
            current_tickets_model=[ current_ticket,new_ticket]
            
            current_ticket_acc,_ = validate(val_loader, current_ticket, criterion) 
            new_ticket_acc,_ = validate(val_loader, new_ticket, criterion) 
            
            
            current_ticket_ind=copy.deepcopy(to_pend_ind)
            current_ticket_ind.append(i)
                
            

            print ("================")

            print ("search inex",i,"current search tickets len",len(current_ticket_ind),new_indx[np.array(current_ticket_ind)])

            
            
            
            
            moving_value=args.interpolation_value_list
            # init model 
            best_moving_acc=0
            best_interpolation=0

            
            print ("beging search interpolate value")
            for i in range(len(moving_value)):
                method=interpolate_LTs(moving_value[i])
                
                interpolation_model = setup_model(args)
                interpolation_model=interpolation_model.cuda()
                params,interpolation_flag = method(current_tickets_model,range(len(current_tickets_model)))
                set_model_params(interpolation_model, params)



                ### prune
            
                if density!=None:
    
                    weight_abs = []

                    for name, weight in interpolation_model.named_parameters():
                        if name not in current_mask: continue
                        weight_abs.append(torch.abs(weight))

                    # Gather all scores in a single vector and normalise
                    all_scores = torch.cat([torch.flatten(x) for x in weight_abs])
                    num_params_to_keep = int(len(all_scores) * density)

                    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
                    acceptable_score = threshold[-1]


                    for name, weight in interpolation_model.named_parameters():
                        if name not in current_mask: continue
                        current_mask[name][:] = ((torch.abs(weight)) >= acceptable_score).float()


                    for name, tensor in interpolation_model.named_parameters():
                        if name in current_mask:
                            tensor.data = tensor.data * current_mask[name]



        #        "update_bn"

                torch.optim.swa_utils.update_bn(train_loader, interpolation_model,"cuda")
        #         print ("update_bn done")
                test_moving_acc,_ = validate(val_loader, interpolation_model, criterion) 
                print('* Test Accurayc = {}'.format(test_moving_acc))
                
                if test_moving_acc > best_moving_acc:
                    best_interpolation_model=copy.deepcopy(interpolation_model)
                    
                    best_moving_acc=test_moving_acc
                    best_interpolation=moving_value[i]
                
                    print ("best interpolation",best_interpolation,"at",best_moving_acc)
                
                
                print ("\n")
                

            if best_moving_acc > best_acc:
                best_acc=best_moving_acc
                print ("best_acc",best_moving_acc)

                current_ticket=copy.deepcopy(best_interpolation_model)
                
                to_pend_ind=current_ticket_ind
                best_interpolation_all.append(best_interpolation)
            
            check_sparsity(best_interpolation_model)
            


            
            
        best_acc,_ = validate(test_loader, current_ticket, criterion) 
        print('*** LTH Accurayc = {}'.format(lth_acc))
        print ("best interpolation acc",best_acc)
        print ("best interpolation value",best_interpolation_all)
        print ("best models",new_indx[np.array(to_pend_ind)])
        print ("**************")



    ## average
    for i in range(1,len(LTs_solutions)+1):
        
        use_LTs_solutions=LTs_solutions[:i]
        




        if args.interpolate_method=="interpolate_ema":
        
                
            new_indx=[]
            start=i-1
            skip_num=1
            new_indx.append(start)
            for _ in range(len(LTs_solutions)):
            #     print ("skip_num",skip_num,i,i-skip_num,i+skip_num)

                if start-skip_num>=0:new_indx.append(start-skip_num)
                if start+skip_num<len(LTs_solutions):new_indx.append(start+skip_num)

                skip_num+=1
                
            print ("new_indx",new_indx)

            use_LTs_solutions=[LTs_solutions[i] for i in new_indx]
            
            
            


            interpolate_ema(use_LTs_solutions,start,new_indx)



        if args.interpolate_method=="interpolate_swa":
    
                
            new_indx=[]
            start=i-1
            skip_num=1
            new_indx.append(start)
            for _ in range(len(LTs_solutions)):
            #     print ("skip_num",skip_num,i,i-skip_num,i+skip_num)

                if start-skip_num>=0:new_indx.append(start-skip_num)
                if start+skip_num<len(LTs_solutions):new_indx.append(start+skip_num)

                skip_num+=1
                
            print ("new_indx",new_indx)

            use_LTs_solutions=[LTs_solutions[i] for i in new_indx]
            
            
            


            interpolate_swa(use_LTs_solutions,start,new_indx)

        if args.interpolate_method=="Lottery_pools":

                
            new_indx=[]
            start=i-1
            skip_num=1
            new_indx.append(start)
            for _ in range(len(LTs_solutions)):
            #     print ("skip_num",skip_num,i,i-skip_num,i+skip_num)

                if start-skip_num>=0:new_indx.append(start-skip_num)
                if start+skip_num<len(LTs_solutions):new_indx.append(start+skip_num)

                skip_num+=1
                
            print ("new_indx",new_indx)

            use_LTs_solutions=[LTs_solutions[i] for i in new_indx]
            
            


            Lottery_pools(use_LTs_solutions,start,new_indx)

                    



        if args.interpolate_method=="liner_inter" : 


            Density=[]
            for i in range(len(sparsity_list)):
                if sparsity_list[i]==None:
                    Density.append(100)
                else:
                    Density.append(np.round(sparsity_list[i],2))

            new_indx=np.array(range(len(LTs_solutions)))
            all_acc,all_loss=liner_inter(LTs_solutions,new_indx)



            break 





if __name__ == '__main__':
    main()
            
