import argparse
import os
import shutil
import time
import random
import math
import copy
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from colorama import Fore
import sys
import re

from collections import OrderedDict


import math

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

# import resnet as models




from  models.imagenet_resnet_core import Model
from  models import bn_initializers, initializers




def add_parser_arguments(parser):
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--save', default='save/default-{}'.format(time.time()),
                           type=str, metavar='SAVE',
                           help='path to the experiment logging directory'
                                '(default: save/debug)')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='imagenet_resnet_18',
                        help='model architecture')

    parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                        help='number of data loading workers (default: 5)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save_dir', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--prof', dest='prof', action='store_true',
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('--master_port', '-master_port', default=10000, type=int,
              help="Port of master for torch.distributed training.")
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument('--seed', default=17, type=int,
                        help='random seed used for np and pytorch')

    parser.add_argument('--update_bn', action='store_true',
                        help='whether update bn layer')




    ##################################### Lottery Pools #################################################
    parser.add_argument('--search_num', default=11, type=int, help=' the count of candidate lotter pools for interpolation')
    parser.add_argument('--EMA_value', default=0.9, type=float, help='EMA factor for interpolation')
    parser.add_argument('--interpolate_method',  type=str , default="Lottery_pools", choices=['Lottery_pools','interpolate_ema', 'interpolate_swa'], help='interpolate_LTs method')
    parser.add_argument('--interpolation_value_list', type=float, nargs='+',help="the  candidate coefficient pools for interpolation")




def main():

    logger = PrintLogger

    main_interpolation(args, logger)


def init_fn(w):
    initializers.kaiming_uniform(w)
    bn_initializers.uniform(w)

def main_interpolation(args, logger_cls):
    exp_start_time = time.time()
    global best_prec1
    best_prec1 = 0

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        os.environ['MASTER_PORT'] = str(args.master_port)
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()


    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)
    else:
        def _worker_init_fn(id):
            pass

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."




    train_loader = get_train_loader(args.data, args.batch_size, workers=args.workers, _worker_init_fn=_worker_init_fn)
    train_loader_len = len(train_loader)



    val_loader = get_val_loader(args.data, args.batch_size, workers=args.workers, _worker_init_fn=_worker_init_fn)
    val_loader_len = len(val_loader)

    test_loader = get_test_loader(args.data, args.batch_size, workers=args.workers, _worker_init_fn=_worker_init_fn)
    test_loader_len = len(test_loader)





    logger = logger_cls(train_loader_len, val_loader_len, args)

    Lottery_interpolation(args, train_loader, val_loader,test_loader,
            args.fp16, logger, prof=args.prof)

    exp_duration = time.time() - exp_start_time
    logger.experiment_timer(exp_duration)
    logger.end_callback()
    print("Good job, all done!")






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




def setup_model(args):
    
        
    # init model 
    ###settings
    cuda=True
    fp16 = args.fp16
    arch=args.arch
    distributed = args.distributed

    model = Model.get_model_from_name(arch, init_fn)

    if cuda:
        model = model.cuda()
    if fp16:
        model = network_to_half(model)
    if distributed:
        model = DDP(model)
 
    return model



def load_check_point(args,level):

        
    # init model 
    ###settings
    cuda=True
    fp16 = args.fp16
    arch=args.arch
    distributed = args.distributed

    model = Model.get_model_from_name(arch, init_fn)

    if cuda:
        model = model.cuda()
    if fp16:
        model = network_to_half(model)
    if distributed:
        model = DDP(model)
 

    checkpoint=torch.load(args.save_dir+"/level_"+str(level)+"/main/model_ep90_it0.pth", map_location = lambda storage, loc: storage.cuda(args.gpu))


    if distributed:
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = "module."+k # add `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)


    loaded_mask=torch.load(args.save_dir+"/level_"+str(level)+"/main/mask.pth", map_location = lambda storage, loc: storage.cuda(args.gpu))

    if distributed:

        new_loaded_mask = OrderedDict()
        for k, v in loaded_mask.items():
            name = "module."+k # add `module.`
            new_loaded_mask[name] = v
        loaded_mask=new_loaded_mask



    for name, tensor in model.named_parameters():
        if name in loaded_mask:
            tensor.data = tensor.data*loaded_mask[name]

    total_size = 0
    for name, weight in model.named_parameters():
        if name in loaded_mask:
            total_size += weight.numel()

    sparse_size = 0
    for name, weight in model.named_parameters():
        if name in loaded_mask:
            sparse_size += (weight != 0).sum().int().item()

    density=sparse_size / total_size


    return model,density




def Lottery_interpolation(args, train_loader, val_loader, test_loader,fp16, logger, prof = False):



    ### ("get LTH solutions")
    print ("=====================================")
    print ("get LTH solutions")
    LTs_solutions=[]
    sparsity_list=[]
    ori_acc=[]

    rangelist=range(args.search_num)
    



    for level in rangelist:   


        model,density=load_check_point(args,level)


        print(' Density level {0}'.format(density))

        sparsity_list.append(density)


        val_acc=validate(test_loader, model, fp16, logger, prof = prof)
        ori_acc.append(val_acc)

        print ("*******")
        LTs_solutions.append(level)





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
        
        current_ticket,_=load_check_point(args,current_ticket)

        to_pend_ind=[0]
        best_interpolation_all=[]

    #     for i in range(1,len(use_LTs_solutions)): 


        for i in range(1,min(len(use_LTs_solutions),args.search_num)  ):  
            
            
            new_ticket=use_LTs_solutions[i]
            
            new_ticket, _=load_check_point(args,new_ticket)
            
            current_tickets_model=[ current_ticket,new_ticket]




            current_ticket_acc=validate(test_loader, current_ticket, fp16, logger, prof = prof)
            new_ticket_acc=validate(test_loader, new_ticket, fp16, logger, prof = prof)
            
            
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



                
                torch.optim.swa_utils.update_bn(prefetched_loader(train_loader,fp16), interpolation_model,"cuda")
 
                test_moving_acc=validate(test_loader, interpolation_model, fp16, logger, prof = prof)
                print('* Test Accurayc = {}'.format(test_moving_acc))

                if test_moving_acc > best_moving_acc:
    
                    if args.distributed:
                        best_interpolation_model = get_model_params(interpolation_model) 
                    else:
                        best_interpolation_model=copy.deepcopy(interpolation_model)
                    
                    best_moving_acc=test_moving_acc
                    best_interpolation=moving_value[i]
                
                    print ("interpolation ratio",best_interpolation,"acc",best_moving_acc)
                
                print ("\n")
            

            best_acc=best_moving_acc
            print ("best_acc",best_moving_acc)


            if args.distributed:
                current_ticket = setup_model(args)
                set_model_params(current_ticket, best_interpolation_model)

            else:
                current_ticket=copy.deepcopy(best_interpolation_model)


            to_pend_ind=current_ticket_ind
            best_interpolation_all.append(best_interpolation)
        


            
            
            
            
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
        current_ticket,_=load_check_point(args,current_ticket)        
        
        to_pend_ind=[0]
        best_interpolation_all=[]

    #     for i in range(1,len(use_LTs_solutions)): 


        for i in range(1,min(len(use_LTs_solutions),args.search_num)  ):  
            
            
            new_ticket=use_LTs_solutions[i]
            new_ticket, _=load_check_point(args,new_ticket)
            
            
            current_tickets_model=[ current_ticket,new_ticket]
            
            current_ticket_acc=validate(test_loader, current_ticket, fp16, logger, prof = prof)
            new_ticket_acc=validate(test_loader, new_ticket, fp16, logger, prof = prof)
            
            
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




                
                torch.optim.swa_utils.update_bn(prefetched_loader(train_loader,fp16), interpolation_model,"cuda")
    
                test_moving_acc=validate(test_loader, interpolation_model, fp16, logger, prof = prof)
                print('* Test Accurayc = {}'.format(test_moving_acc))
                
                if test_moving_acc > best_moving_acc:

                    if args.distributed:
                        best_interpolation_model = get_model_params(interpolation_model) 
                    else:
                        best_interpolation_model=copy.deepcopy(interpolation_model)
                    
                    best_moving_acc=test_moving_acc
                    best_interpolation=moving_value[i]
                
                    print ("interpolation ratio",best_interpolation,"acc",best_moving_acc)
                
                print ("\n")
            

            best_acc=best_moving_acc
            print ("best_acc",best_moving_acc)


            if args.distributed:
                current_ticket = setup_model(args)
                set_model_params(current_ticket, best_interpolation_model)

            else:
                current_ticket=copy.deepcopy(best_interpolation_model)


            
            to_pend_ind=current_ticket_ind
            best_interpolation_all.append(best_interpolation)
        

            


            
            
            
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
        current_ticket,_=load_check_point(args,current_ticket)    
        
        to_pend_ind=[0]
        best_interpolation_all=[]

    #     for i in range(1,len(use_LTs_solutions)): 


        for i in range(1,min(len(use_LTs_solutions),args.search_num)  ):  
            
            
            new_ticket=use_LTs_solutions[i]
            new_ticket,_, _=load_check_point(args,new_ticket)    
            
            
            current_tickets_model=[ current_ticket,new_ticket]
            
            current_ticket_acc=validate(val_loader, current_ticket, fp16, logger, prof = prof)
            new_ticket_acc=validate(val_loader, new_ticket, fp16, logger, prof = prof)
            
            
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



                torch.optim.swa_utils.update_bn(prefetched_loader(train_loader,fp16), interpolation_model,"cuda")
 


                test_moving_acc =validate(val_loader, interpolation_model, fp16, logger, prof = prof) 

                
                print('* Cal Accurayc = {}'.format(test_moving_acc))
                
                if test_moving_acc > best_moving_acc:

                    if args.distributed:
                        best_interpolation_model = get_model_params(interpolation_model) 
                    else:
                        best_interpolation_model=copy.deepcopy(interpolation_model)
                    
                    best_moving_acc=test_moving_acc
                    best_interpolation=moving_value[i]
                
                    print ("best interpolation",best_interpolation,"at",best_moving_acc)
                
                
                print ("\n")
                

            if best_moving_acc > best_acc:
                best_acc=best_moving_acc
                print ("best_acc",best_moving_acc)

                if args.distributed:
                    current_ticket = setup_model(args)
                    set_model_params(current_ticket, best_interpolation_model)

                else:
                    current_ticket=copy.deepcopy(best_interpolation_model)
                    
                to_pend_ind=current_ticket_ind
                best_interpolation_all.append(best_interpolation)
            

   
            
        best_acc,_ = validate(test_loader, current_ticket, criterion) 
        print('*** LTH Accurayc = {}'.format(lth_acc))
        print ("best interpolation_acc",best_acc)
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

                    


# }}}

# Data Loading functions {{{
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        # tens = torch.from_numpy(nump_array)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        nump_array_copy = np.copy(nump_array)
        tensor[i] += torch.from_numpy(nump_array_copy)

    return tensor, targets


def prefetched_loader(loader, fp16):
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
    if fp16:
        mean = mean.half()
        std = std.half()

    stream = torch.cuda.Stream()
    first = True

    for next_input, next_target in loader:
        with torch.cuda.stream(stream):
            next_input = next_input.cuda(non_blocking=True)
            next_target = next_target.cuda(non_blocking=True)
            if fp16:
                next_input = next_input.half()
            else:
                next_input = next_input.float()
            next_input = next_input.sub_(mean).div_(std)

        if not first:
            yield input, target
        else:
            first = False

        torch.cuda.current_stream().wait_stream(stream)
        input = next_input
        target = next_target

    yield input, target


def get_train_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(), Too slow
            #normalize,
        ]))

    if torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate, drop_last=True)

    return train_loader

def get_val_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    valdir = os.path.join(data_path, 'val')

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
        collate_fn=fast_collate)

    return val_loader

def get_test_loader(data_path, batch_size, workers=5, _worker_init_fn=None):
    valdir = os.path.join(data_path, 'test')

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, worker_init_fn=_worker_init_fn, pin_memory=True,
        collate_fn=fast_collate)

    return test_loader



def get_val_step(model):
    def _step(input, target):
        input_var = Variable(input)
        target_var = Variable(target)

        with torch.no_grad():

            output = model(input_var)

        prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))

        if torch.distributed.is_initialized():

            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)


        torch.cuda.synchronize()

        return  prec1, prec5

    return _step


def validate(val_loader, model, fp16, logger, prof=False):
    epoch=0
    step = get_val_step(model)

    top1 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(prefetched_loader(val_loader, fp16)):
        data_time = time.time() - end
        if prof:
            if i > 10:
                break

        prec1, prec5 = step(input, target)

        top1.update(to_python_float(prec1), input.size(0))

        logger.val_iter_callback( epoch,i,
                {'size' : input.size(0),
                 'top1' : to_python_float(prec1),
                 'top5' : to_python_float(prec5),
                 'time' : time.time() - end,
                 'data' : data_time})

        end = time.time()

    logger.val_epoch_callback(epoch)

    return top1.avg

# }}}

# Logging {{{


class EpochLogger(object):
    def __init__(self, name, total_iterations, args):
        self.name = name
        self.args = args
        self.print_freq = args.print_freq
        self.total_iterations = total_iterations
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        self.time = AverageMeter()
        self.data = AverageMeter()

    def iter_callback(self, epoch, iteration, d):
        self.top1.update(d['top1'], d['size'])
        self.top5.update(d['top5'], d['size'])
        self.time.update(d['time'], d['size'])
        self.data.update(d['data'], d['size'])

        if iteration % self.print_freq == 0:
            print('{0}:\t{1} [{2}/{3}]\t'
                  'Time {time.val:.3f} ({time.avg:.3f})\t'
                  'Data time {data.val:.3f} ({data.avg:.3f})\t'
                  'Speed {4:.3f} ({5:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  self.name, epoch, iteration, self.total_iterations,
                  self.args.world_size * self.args.batch_size / self.time.val,
                  self.args.world_size * self.args.batch_size / self.time.avg,
                  time=self.time,
                  data=self.data,
   
                  top1=self.top1,
                  top5=self.top5))

    def epoch_callback(self, epoch):
        print('{0} epoch {1} summary:\t'
              'Time {time.avg:.3f}\t'
              'Data time {data.avg:.3f}\t'
              'Speed {2:.3f}\t'
              'Prec@1 {top1.avg:.3f}\t'
              'Prec@5 {top5.avg:.3f}'.format(
              self.name, epoch,
              self.args.world_size * self.args.batch_size / self.time.avg,
              time=self.time, data=self.data,
            top1=self.top1, top5=self.top5))

        self.top1.reset()
        self.top5.reset()

        self.time.reset()
        self.data.reset()


class PrintLogger(object):
    def __init__(self, train_iterations, val_iterations, args):
        self.train_logger = EpochLogger("Train", train_iterations, args)
        self.val_logger = EpochLogger("Eval", val_iterations, args)

    def train_iter_callback(self, epoch, iteration, d):
        self.train_logger.iter_callback(epoch, iteration, d)

    def train_epoch_callback(self, epoch):
        self.train_logger.epoch_callback(epoch)
        
    def val_iter_callback(self, epoch, iteration, d):
        self.val_logger.iter_callback(epoch, iteration, d)

    def val_epoch_callback(self, epoch):
        self.val_logger.epoch_callback(epoch)
        
    def experiment_timer(self, exp_duration):
        print("Experiment took {} seconds".format(exp_duration))

    def end_callback(self):
        pass


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

# }}}


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        print("SAVING")

        torch.save(state, filename)

def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function








def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()
    return rt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    update_iter = 0
    add_parser_arguments(parser)
    args = parser.parse_args()
    cudnn.benchmark = True
    print(args)
    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    main()
