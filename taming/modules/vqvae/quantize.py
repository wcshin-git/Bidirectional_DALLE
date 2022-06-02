import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta, ortho_reg_coef, force_emb_len_1=False, distance_measure='L2'):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.ortho_reg_coef = ortho_reg_coef
        self.force_emb_len_1 = force_emb_len_1
        self.distance_measure = distance_measure
        assert distance_measure in ['L2', 'cos_sim'], "distance_measure MUST be in ['L2', 'cos_sim']"

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        if self.distance_measure == 'L2':
            # L2 distance
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.matmul(z_flattened, self.embedding.weight.t())
        
        elif self.distance_measure == 'cos_sim':
            ## cosine similarity
            # first normalize to 1
            z_norm = F.normalize(z_flattened, p=2, dim=1)
            weight_norm = F.normalize(self.embedding.weight, p=2, dim=1)
            # compute inner product
            d = -torch.mm(z_norm, weight_norm.transpose(0, 1))

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
            torch.mean((z_q - z.detach()) ** 2)
        
        # ortho reg term
        if self.force_emb_len_1 == False:
            emb_weight_after_norm = nn.functional.normalize(self.embedding.weight, p=2, dim=1)
            diff = torch.mm(emb_weight_after_norm, torch.transpose(emb_weight_after_norm, 0, 1)) - torch.eye(self.n_e, self.n_e).type_as(emb_weight_after_norm)
            ortho_reg_term = self.ortho_reg_coef * torch.sum(diff**2) / (diff.size(0)**2)
        
        elif self.force_emb_len_1 == True:
            diff = torch.mm(self.embedding.weight, torch.transpose(self.embedding.weight, 0, 1)) - torch.eye(self.n_e, self.n_e).type_as(self.embedding.weight)
            ortho_reg_term = self.ortho_reg_coef * torch.sum(diff**2) / (diff.size(0)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, (loss, ortho_reg_term), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
