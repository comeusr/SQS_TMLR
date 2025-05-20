import wandb
from composer import Callback, State, Logger, Event
from SQS.modeling.DGMS import DGMSConv
import torch.nn as nn
import torch


def check_total_zero(x):
    with torch.no_grad():
        return x.eq(0.0).float().sum().item()


def check_total_weights(x):
    with torch.no_grad():
        return x.numel()


class Sparsity(Callback):

    def run_event(self, event:Event, state:State, logger:Logger):
        if event == Event.EPOCH_START:
            print("Try to print Module Name")
            for name, m in state.model.named_modules():
                print(name)
            

class EpochMonitor(Callback):

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.EPOCH_END:
            print(f'Epoch: {state.timestamp.epoch}')
            tot_numel = 0
            tot_zero_numel = 0
            for name, m in state.model.named_modules():
                if isinstance(m, DGMSConv):
                    with torch.no_grad():
                        data = m.sub_distribution.mu.detach().data.cpu().numpy()
                        P_weight = m.get_Pweight()
                        S_weight = m.get_Sweight()
                        Origin_weight = m.weight
                        P_weight_zeros = check_total_zero(P_weight)
                        P_weight_tot = check_total_weights(P_weight)
                        S_weight_zeros = check_total_zero(S_weight)
                        S_weight_tot = check_total_weights(S_weight)
                        Origin_weight_zeros=check_total_zero(Origin_weight)
                        Origin_weight_tot=check_total_weights(Origin_weight)
                        # hist = np.histogram(data)
                        # print(hist)
                        tot_numel += P_weight_tot
                        tot_zero_numel += P_weight_zeros

                        wandb.log({name+"_mu": wandb.Histogram(data)}, commit=False)
                        
                        wandb.log({name+"_P_zeros": P_weight_zeros/P_weight_tot}, commit=False)
                        wandb.log({name+"_S_zeros": S_weight_zeros/S_weight_tot}, commit=False)
                    wandb.log({'Total Sparsity': tot_zero_numel/tot_numel}, commit=False)

                    

                elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                    total_zero = check_total_zero(m.weight)
                    total_weight = check_total_weights(m.weight)
                    wandb.log({name+"_sparsity": total_zero/total_weight}, commit=False)
            


