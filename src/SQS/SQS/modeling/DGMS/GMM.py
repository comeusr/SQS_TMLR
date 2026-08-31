# DGMS GM Sub-distribution implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import SQS.config as cfg

from SQS.utils.misc import cluster_weights, get_device, cluster_weights_sparsity
from SQS.utils.utils import get_distribution
from SQS.utils.interval_mapping import interval_mapping, reconstruct
import time

DEVICE = get_device()


def gumbel_max_sample(logits: torch.Tensor) -> torch.Tensor:
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))  # Gumbel(0,1) noise
    return torch.argmax(logits + gumbel_noise, dim=-1)



class GaussianMixtureModel(nn.Module):
    """Concrete GMM for sub-distribution approximation.
    """
    def __init__(self, num_components, init_weights, temperature=0.01, B=11, init_method="k-means", init_sigma=3):
        super(GaussianMixtureModel, self).__init__()
        self.num_components = num_components
        self.temperature = temperature

        # print("Initializing GMM Parameters.")
        shape = init_weights.shape
        self.mu_zero = torch.tensor([0.0], device=init_weights.device).float()
        self.sigma_zero =nn.Parameter(data=torch.tensor([0.0], device=init_weights.device).float(), requires_grad=(cfg.METHOD == "DGMS"))

        self.pi_k, self.mu, self.sigma = \
                    nn.Parameter(data=torch.ones(self.num_components, device=DEVICE), requires_grad=True), \
                    nn.Parameter(data=torch.ones(self.num_components, device=DEVICE), requires_grad=True), \
                    nn.Parameter(data=torch.ones(self.num_components, device=DEVICE), requires_grad=(cfg.METHOD == "DGMS"))

        self.temperature = torch.tensor([self.temperature], device=DEVICE)
        self.pruning_parameter = nn.Parameter(data=torch.ones_like(init_weights, device=DEVICE), requires_grad=(cfg.METHOD == "SQS"))

        print("Init_method", init_method)
        self.params_initialization(init_weights, init_method)
        self.prune = cfg.PRUNE
        self.method = cfg.METHOD
        self.mask = (init_weights.abs()< 0.0)

        if cfg.PRUNE:
            self.init_sigma = init_sigma
        # Reduced dimension
        self.B = B

        
        start_time = time.time()
        print("Starting Grouping")
        self.bin_indices, self.nums = interval_mapping(init_weights, self.B, DEVICE)
        end_time = time.time()
        print("Grouping completed in {} seconds".format(end_time - start_time))

        if self.nums.isnan().any():
            print("GMM found nan in self.nums after interval mapping {}".format(self.nums))

    def params_initialization(self, init_weights, method='k-means'):
        if not cfg.PRUNE:
            """ Initialization of GMM parameters using k-means algorithm. """
            self.mu_zero.requires_grad = True
            self.pi_k.data, self.mu.data, self.sigma.data = \
                    torch.ones(self.num_components-1, device=DEVICE), \
                    torch.ones(self.num_components-1, device=DEVICE), \
                    torch.ones(self.num_components-1, device=DEVICE)
            if method == 'k-means':
                print("Using k-means for GMM initialization")
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            elif method == "quantile":
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            elif method == 'empirical':
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
                sigma_init, _sigma_zero = torch.ones_like(sigma_init).mul(0.01), torch.ones_like(torch.tensor([_sigma_zero])).mul(0.01)

            if sigma_init.isnan().any() or sigma_init.eq(0.0).any():
                print('-'*50+"Sigma init is nan or zero"+"-"*50)
                print('-'*50+"Old Sigma init {}".format(sigma_init)+"-"*50)
                # Create a mask for valid elements (nonzero and non-NaN)
                valid_mask = (sigma_init != 0) & (~torch.isnan(sigma_init)) & (torch.isfinite(sigma_init))

                # Extract valid elements
                valid_elements = sigma_init[valid_mask]
                
                smallest_valid = torch.min(valid_elements)
                largest_valid = torch.max(valid_elements)
                # Replace zeros and NaNs with the smallest valid elements
                sigma_init[sigma_init == 0] = smallest_valid
                sigma_init[torch.isnan(sigma_init)] = smallest_valid
                sigma_init[~torch.isfinite(sigma_init)] = largest_valid
                print('-'*50+"New Sigma init {}".format(sigma_init)+"-"*50)

            self.mu = nn.Parameter(data=torch.mul(self.mu, initial_region_saliency.flatten()))
            self.pi_k = nn.Parameter(data=torch.mul(self.pi_k, pi_init)).float()
            self.pi_zero = nn.Parameter(data=torch.tensor([pi_zero_init], device=init_weights.device)).float()
            self.sigma_zero = nn.Parameter(data=torch.tensor([_sigma_zero], device=init_weights.device)).float()
            self.sigma = nn.Parameter(data=torch.mul(self.sigma, sigma_init)).float()
            self.temperature = nn.Parameter(data=torch.tensor([self.temperature], device=init_weights.device), requires_grad=False)
        else:
            """ Intialization of GMM + Pruning parameters using k-means"""
            print("Method", method)
            if method == 'k-means':
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
            elif method == "quantile":
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
            elif method == 'empirical':
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
                sigma_init = torch.ones_like(sigma_init).mul(0.01)
                
            if sigma_init.isnan().any() or sigma_init.eq(0.0).any():
                print('-'*50+"Sigma init is nan or zero"+"-"*50)
                print('-'*50+"Old Sigma init {}".format(sigma_init)+"-"*50)
                # Create a mask for valid elements (nonzero and non-NaN)
                valid_mask = (sigma_init != 0) & (~torch.isnan(sigma_init)) & (torch.isfinite(sigma_init))

                # Extract valid elements
                valid_elements = sigma_init[valid_mask]
                
                smallest_valid = torch.min(valid_elements)
                largest_valid = torch.max(valid_elements)
                # Replace zeros and NaNs with the smallest valid elements
                sigma_init[sigma_init == 0] = smallest_valid
                sigma_init[torch.isnan(sigma_init)] = smallest_valid
                sigma_init[~torch.isfinite(sigma_init)] = largest_valid
                print('-'*50+"New Sigma init {}".format(sigma_init)+"-"*50)

            self.mu.data = torch.mul(self.mu, initial_region_saliency.flatten())
            self.pi_k.data = torch.mul(self.pi_k, pi_init).float()
            self.sigma.data = torch.mul(self.sigma, sigma_init).float()

            self.temperature = torch.tensor([self.temperature], device=DEVICE)
            self.pruning_parameter.data = 5*cfg.PRUNE_SCALE*torch.ones_like(init_weights, device=DEVICE)

    def gaussian_mixing_regularization(self):
        
        if not cfg.PRUNE:
            pi_tmp = torch.cat([self.pi_zero, self.pi_k], dim=-1).abs()
            res = torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1))
            return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1))
        else:
            pi_tmp = self.pi_k.abs()
            return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1))

    def Normal_pdf(self, x, _pi, mu, sigma):
        """ Standard Normal Distribution PDF. """
        pow2 = torch.pow(x - mu, 2)
        
        sigma = sigma.to(torch.float32)
        pdf = torch.mul(torch.reciprocal(torch.sqrt(torch.mul( \
                torch.tensor([2 * math.pi], device=DEVICE), (sigma**2)))), \
                    torch.exp(-torch.div(pow2, 2 * sigma**2))).mul(_pi)

        return pdf


    def GMM_region_responsibility(self, weights):
        pi_normalized = self.gaussian_mixing_regularization()

        O = get_distribution(self.nums, self.mu, self.num_components, pi_normalized, self.sigma, self.sigma_zero, self.method, DEVICE)
        O = torch.div(O, O.sum(dim=0) + cfg.EPS)

        # R3: at eval, optionally raise the softmax temperature (EVAL_TEMP_SCALE>1) so the
        # responsibilities soften and multinomial sampling produces diverse quantizations ->
        # Bayesian averaging over average_num copies actually reduces variance. Training uses
        # the sharp temperature unchanged (IS_TRAIN=True -> scale 1.0).
        temp_eff = self.temperature * (cfg.EVAL_TEMP_SCALE if not cfg.IS_TRAIN else 1.0)
        temp = F.softmax(O / temp_eff, dim=0).T
        return temp



    def forward(self, weights, train=True):
        if cfg.METHOD == "DGMS":
            if train:
                # soft mask generalized pruning during training
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())

                Sweight = reconstruct(weights.shape, torch.mul(self.region_belonging[:,0], 0.), self.bin_indices, DEVICE)+\
                    reconstruct(weights.shape, self.region_belonging[:,1:]@self.mu.unsqueeze(1), self.bin_indices, DEVICE)

                return Sweight.view(weights.size())
            else:
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())
                max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
                mask_w = torch.zeros_like(self.region_belonging).scatter_(dim=0, index=max_index, value=1.)

                Pweight = reconstruct(weights.shape, mask_w[:,1:]@self.mu.unsqueeze(1), self.bin_indices, DEVICE)
                return Pweight.view(weights.size())
        else:
            if train:
                region_belonging = self.GMM_region_responsibility(weights.flatten())

                if cfg.PRIOR == 'spike_slab':
                    temp = reconstruct(weights.shape, region_belonging@self.mu, self.bin_indices, DEVICE)
                    Sweight =  temp* F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE)
                    self.sweight_cache = torch.abs(temp)*F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE)
                    # Activation-aware (Wanda-style) saliency: scale magnitude by the
                    # input-channel activation norm ||X_j|| so the pruner keeps weights
                    # that are large AND actually used. No-op unless calibrated.
                    if getattr(self, 'act_norm', None) is not None:
                        self.sweight_cache = self.sweight_cache * self.act_norm.flatten().to(self.sweight_cache.dtype)
                else:
                    Sweight = torch.mul(region_belonging, self.mu.unsqueeze(1)).sum(dim=0)* F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE) \
                            + (1-F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE))*torch.randn_like(weights.flatten())



                return Sweight.view(weights.size())
            else:
                region_belonging = self.GMM_region_responsibility(weights.flatten())
                
                if cfg.SAMPLE:
                    max_index = region_belonging.multinomial(num_samples=1)
                else:
                    max_index = torch.argmax(region_belonging, dim=1).unsqueeze(1)

                
                mask_w = torch.zeros_like(region_belonging).scatter_(dim=1, index=max_index, value=1.)


                Pweight = reconstruct(weights.size(), mask_w@self.mu, self.bin_indices, DEVICE)

                Pweight = Pweight.view(weights.size())

                # Realize spike-and-slab sparsity at inference. The train branch gates the
                # weight by sigmoid(pruning_parameter/PRUNE_SCALE); pruned entries are driven
                # to pruning_parameter<0 (gate<0.5). Without applying the hard gate here the
                # learned sparsity is silently discarded at eval, so measured non-zero stays
                # ~100% (NZ ratio 1.0) even though the sparsity schedule ran. Apply it so the
                # evaluated model is actually sparse and check_total_zero sees exact zeros.
                if cfg.METHOD == "SQS" and cfg.PRIOR == "spike_slab":
                    # Match the TRAIN branch, which gates every weight by the SOFT
                    # sigmoid(pruning_parameter/PRUNE_SCALE) (line ~196). The KL/spike-slab
                    # prior drives KEPT weights' pruning_parameter down from its init (5*scale
                    # -> gate~0.993) toward 0 (gate->0.5), so the network trains with kept
                    # weights scaled by ~0.5-0.99. Using a hard 1.0 gate at eval made kept
                    # weights up to ~2x too large -> activation blow-up -> ResNet collapsed to
                    # ~1% at prune onset. Applying the same soft gate here aligns eval with
                    # train; the hard (~mask) still forces the pruned entries to EXACT zero so
                    # check_total_zero measures the intended sparsity.
                    soft = torch.sigmoid(self.pruning_parameter.flatten() / cfg.PRUNE_SCALE).view(weights.size())
                    gate = (~self.mask).to(Pweight.dtype).view(weights.size())
                    Pweight = Pweight * soft * gate

                return Pweight

def gmm_approximation(num_components, init_weights, temperature=0.5, B=11, init_method='k-means', sigma=3) -> GaussianMixtureModel:
    return GaussianMixtureModel(num_components, init_weights, temperature, B, init_method, sigma)