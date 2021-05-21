import numpy as np
import torch
import pickle
import pdb
from utils.utils import *
import copy
import os
# import syft as sy
from datasets.dataset_generic import save_splits
from sklearn.metrics import roc_auc_score
from models.model_attention_mil import MIL_Attention_fc_surv
from utils.fl_utils import sync_models, federated_averging

from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation metric decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train_fl_surv(datasets, cur, args):
    """   
        train for a single fold
    """
    # number of institutions
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)
    
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss()
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss()
    else:
        raise NotImplementedError
   
    print('Done!')


    print('\nInit train/val/test splits...', end=' ')
    train_splits, val_split, test_split = datasets
    num_insti = len(train_splits)
    print('Done!')
    for idx in range(num_insti):
        print("Worker_{} Training on {} samples".format(idx,len(train_splits[idx])))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})

        model = MIL_Attention_fc_surv(**model_dict)
        worker_models =[MIL_Attention_fc_surv(**model_dict) for idx in range(num_insti)]

    else:
        raise NotImplementedError
    print('Done!')
    
    sync_models(model, worker_models)   
    device_counts = torch.cuda.device_count()
    if device_counts > 1:
        device_ids = [idx % device_counts for idx in range(num_insti)]
    else:
        device_ids = [0]*num_insti
    
    model.relocate(device_id=0)
    for idx in range(num_insti):
        worker_models[idx].relocate(device_id=device_ids[idx])

    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    worker_optims = [get_optim(worker_models[i], args) for i in range(num_insti)]
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loaders = []
    for idx in range(num_insti):
        train_loaders.append(get_split_loader(train_splits[idx], training=True, testing = args.testing, 
                                              weighted = False, task_type = 'survival'))
    val_loader = get_split_loader(val_split, task_type = 'survival', testing = args.testing)
    test_loader = get_split_loader(test_split, task_type = 'survival', testing = args.testing)

    if args.weighted_fl_avg:
        weights = np.array([len(train_loader) for train_loader in train_loaders]) 
        weights = weights / weights.sum()
    else:
        weights = None

    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch= 35, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):       
        train_loop_fl_surv(epoch, model, worker_models, train_loaders, worker_optims, 
                     args.n_classes, writer, loss_fn)
        if (epoch + 1) % args.E == 0:
            model, worker_models = federated_averging(model, worker_models, args.noise_level, weights)
            sync_models(model, worker_models)  
        stop = validate_surv(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_c_index = summary_surv(model, val_loader, args.n_classes)
    print('Val c-index: {:.4f}'.format(val_c_index))

    results_dict, test_c_index = summary_surv(model, test_loader, args.n_classes)
    print('Test c-index: {:.4f}'.format(test_c_index))

    if writer:
        writer.add_scalar('final/val_c_index', val_c_index, 0)
        writer.add_scalar('final/test_c_index', test_c_index, 0)
    
    writer.close()
    return results_dict, test_c_index, val_c_index

def train_surv(datasets, cur, args):
    """   
        train for a single fold
    """
    # number of institutions
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    from tensorboardX import SummaryWriter
    writer = SummaryWriter(writer_dir, flush_secs=15)

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss()
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss()
    else:
        raise NotImplementedError
   
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type =='attention_mil':
        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})

        model = MIL_Attention_fc_surv(**model_dict)
    else:
        raise NotImplementedError

    model.relocate()
    print_network(model)
    print('Done!')


    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    print('Done!')
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
                                    weighted = args.weighted_sample, task_type = 'survival')
    val_loader = get_split_loader(val_split, task_type = 'survival', testing = args.testing)
    test_loader = get_split_loader(test_split, task_type = 'survival', testing = args.testing)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch= 35, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):        
        train_loop_surv(epoch, model, train_loader, optimizer, 
                     args.n_classes, writer, loss_fn)
        stop = validate_surv(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        if stop: 
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_c_index = summary_surv(model, val_loader, args.n_classes)
    print('Val c-index: {:.4f}'.format(val_c_index))

    results_dict, test_c_index = summary_surv(model, test_loader, args.n_classes)
    print('Test c-index: {:.4f}'.format(test_c_index))

    if writer:
        writer.add_scalar('final/val_c_index', val_c_index, 0)
        writer.add_scalar('final/test_c_index', test_c_index, 0)
    
    writer.close()
    return results_dict, test_c_index, val_c_index

def train_loop_fl_surv(epoch, model, worker_models, worker_loaders, worker_optims, n_classes, writer=None, loss_fn=None):
    num_insti = len(worker_models)    
    model.train()
    
    train_loss = 0.

    print('\n')
    total = np.sum(len(worker_loaders[i]) for i in range(num_insti))
    all_risk_scores = np.zeros(total)
    all_censorships = np.zeros(total)
    all_event_times = np.zeros(total)

    print('\n')
    for idx in range(len(worker_loaders)):
        # pdb.set_trace()
        if worker_models[idx].device is not None:
            model_device = worker_models[idx].device
        else:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for batch_idx, (data, label, event_time, c) in enumerate(worker_loaders[idx]):
            data, label, c = data.to(model_device), label.to(model_device), c.to(model_device)
            # logits, Y_prob, Y_hat, _, _ = worker_models[idx](data)

            hazards, survival, Y_hat, _, _ = worker_models[idx](data)

            loss = loss_fn(hazards, label, c, survival)

            loss_value = loss.item()
            train_loss += loss_value

            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            
            if (batch_idx + 1) % 5 == 0:
                print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, '.format(batch_idx, loss_value, label.item(), float(event_time)) +
                      'risk: {:.4f}, bag_size: {}'.format( float(risk), data.size(0)))

            # backward pass
            loss.backward()
            # step
            worker_optims[idx].step()
            worker_optims[idx].zero_grad()

    # calculate loss and error for epoch
    train_loss /= np.sum(len(worker_loaders[i]) for i in range(num_insti))
    # print('model updated: ', torch.abs(model_params - model.classifier.weight).sum())
    try:
        c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    except:
        pdb.set_trace()
    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))

    if writer:
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)

def train_loop_surv(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss = 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    for batch_idx, (data, label, event_time, c) in enumerate(loader):
        data, label, c = data.to(device), label.to(device), c.to(device)

        hazards, survival, Y_hat, _, _ = model(data)

        loss = loss_fn(hazards, label, c, survival)

        loss_value = loss.item()
        risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss += loss_value
        if (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value, label.item(), float(event_time), float(risk), data.size(0)))
        
        # backward pass
        loss.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss /= len(loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss, c_index))

    if writer:
        writer.add_scalar('train/c_index', c_index, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
   
def validate_surv(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data, label, event_time, c) in enumerate(loader):
        data, label, c = data.to(device), label.to(device), c.to(device)
        
        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(data)

        loss = loss_fn(hazards, label, c, survival, alpha=0)
        loss_value = loss.item()

        risk = -torch.sum(survival, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        val_loss += loss.item()

    val_loss /= len(loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

    print('\nVal Set, val_loss: {:.4f}, val c-index: {:.4f}'.format(val_loss, c_index))     

    if early_stopping:
        assert results_dir
        early_stopping(epoch, -c_index, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False

def summary_surv(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label, event_time, c) in enumerate(loader):
        data, label= data.to(device), label.to(device)
        
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            hazards, survival, Y_hat, _, _ = model(data)

        risk = np.asscalar(-torch.sum(survival, dim=1).cpu().numpy())
        event_time = np.asscalar(event_time)
        c = np.asscalar(c)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time, 'censorship': c}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index
