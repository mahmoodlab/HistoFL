import torch
import pdb
def sync_models(server_model, worker_models):
    server_params = server_model.state_dict()
    worker_models = [worker_model.load_state_dict(server_params) for worker_model in worker_models]

def federated_averging(model, worker_models, noise_level, weights=None):
    if model.device is not None:
        central_device = model.device
    else:
        central_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    num_insti = len(worker_models)
    if weights is None:
        weights = [1/num_insti for i in range(num_insti)]
    central_params = model.state_dict()
    all_worker_params = [worker_models[idx].state_dict() for idx in range(num_insti)]
    keys = central_params.keys()
    for key in keys:
        if 'labels' in key:
            continue
        else:
            temp = torch.zeros_like(central_params[key])
            for idx in range(num_insti):
                if noise_level > 0 and 'bias' not in key: 
                    noise = noise_level * torch.empty(all_worker_params[idx][key].size()).normal_(mean=0, 
                                                                                                  std=all_worker_params[idx][key].reshape(-1).float().std())
                    temp = temp + weights[idx] * all_worker_params[idx][key].to(central_device) + noise.to(central_device)             
                else:
                    temp = temp + weights[idx] * all_worker_params[idx][key].to(central_device)

            central_params[key] = temp       
    model.load_state_dict(central_params)
    return model, worker_models