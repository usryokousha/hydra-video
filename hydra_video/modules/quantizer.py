import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict


def l2_normalize(x: torch.Tensor, dim=None, epsilon=1e-12) -> torch.Tensor:
    """
    Performs L2 normalization on the input tensor.

    Args:
        x (torch.Tensor): Input tensor to be normalized.
        dim (int, optional): The dimension along which to normalize. Defaults to None.
        epsilon (float, optional): Small value to avoid division by zero. Defaults to 1e-12.

    Returns:
        torch.Tensor: L2 normalized tensor.
    """
    square_sum = torch.sum(x**2, dim=dim, keepdim=True)
    x_inv_norm = torch.rsqrt(torch.clamp(square_sum, min=epsilon))
    return x * x_inv_norm


def squared_euclidean_distance(
    a: torch.Tensor, b: torch.Tensor, b2: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes the pairwise squared Euclidean distance between two sets of vectors.

    Args:
        a (torch.Tensor): First set of vectors, shape (n, d).
        b (torch.Tensor): Second set of vectors, shape (m, d).
        b2 (torch.Tensor, optional): Precomputed squared norm of b. Defaults to None.

    Returns:
        torch.Tensor: Pairwise squared distances, shape (n, m).
    """
    if b2 is None:
        b2 = torch.sum(b.t() ** 2, dim=0, keepdim=True)
    a2 = torch.sum(a**2, dim=1, keepdim=True)
    ab = torch.matmul(a, b.t())
    d = a2 - 2 * ab + b2
    return d


def entropy_loss(
    affinity: torch.Tensor, loss_type: str = "softmax", temperature: float = 1.0
) -> torch.Tensor:
    """
    Calculates the entropy loss for the given affinity matrix.

    Args:
        affinity (torch.Tensor): Affinity matrix.
        loss_type (str, optional): Type of entropy loss ("softmax" or "argmax"). Defaults to "softmax".
        temperature (float, optional): Temperature for scaling the affinity. Defaults to 1.0.

    Returns:
        torch.Tensor: Computed entropy loss.

    Raises:
        ValueError: If an unsupported loss_type is provided.
    """
    flat_affinity = affinity.view(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)

    if loss_type == "softmax":
        target_probs = probs
    elif loss_type == "argmax":
        codes = torch.argmax(flat_affinity, dim=-1)
        onehots = F.one_hot(codes, flat_affinity.shape[-1]).to(flat_affinity.dtype)
        onehots = probs - (probs - onehots).detach()
        target_probs = onehots
    else:
        raise ValueError(f"Entropy loss {loss_type} not supported")

    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


class Quantizer(nn.Module):
    """
    Base class for quantizer modules.
    """

    def forward(self, x: torch.Tensor, **kwargs) -> tuple:
        raise NotImplementedError

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_codebook(self) -> torch.Tensor:
        raise NotImplementedError

    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VectorQuantizer(Quantizer):
    """
    Vector Quantizer implementation.
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        entropy_loss_ratio: float = 0.0,
        entropy_loss_type: str = "softmax",
        entropy_temperature: float = 1.0,
        latent_normalize: bool = False,
    ):
        """
        Initialize the Vector Quantizer.

        Args:
            codebook_size (int): Size of the codebook.
            embedding_dim (int): Dimension of each codebook vector.
            commitment_cost (float, optional): Weight for commitment loss. Defaults to 0.25.
            entropy_loss_ratio (float, optional): Weight for entropy loss. Defaults to 0.0.
            entropy_loss_type (str, optional): Type of entropy loss. Defaults to "softmax".
            entropy_temperature (float, optional): Temperature for entropy calculation. Defaults to 1.0.
        """
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.entropy_loss_ratio = entropy_loss_ratio
        self.entropy_loss_type = entropy_loss_type
        self.entropy_temperature = entropy_temperature
        self.latent_normalize = latent_normalize

        self.codebook = nn.Parameter(
            torch.randn(self.codebook_size, self.embedding_dim)
        )
        nn.init.xavier_uniform_(self.codebook)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the Vector Quantizer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Quantized tensor and a dictionary containing additional information.
        """
        flat_x = x.view(-1, self.embedding_dim)

        with torch.autocast(enabled=False):
            if self.latent_normalize:
                flat_x = l2_normalize(flat_x, dim=-1)
                codebook = l2_normalize(self.codebook, dim=-1)

            distances = squared_euclidean_distance(flat_x, codebook)
            encoding_indices = torch.argmin(distances, dim=1)
            encodings = F.one_hot(encoding_indices, self.codebook_size).float()
            quantized = self.quantize(encodings)

            # Reshape quantized to match input shape
            quantized = quantized.view(x.shape)

            result_dict = dict()
            if self.training:
                commitment_loss = F.mse_loss(quantized.detach(), x)
                codebook_loss = F.mse_loss(quantized, x.detach())

                entropy_loss = 0.0
                if self.entropy_loss_ratio != 0:
                    entropy_loss = (
                        entropy_loss(
                            -distances.view(x.shape[:-1] + (-1,)),
                            loss_type=self.entropy_loss_type,
                            temperature=self.entropy_temperature,
                        )
                        * self.entropy_loss_ratio
                    )

                loss = commitment_loss * self.commitment_cost + codebook_loss + entropy_loss

                # Straight-through estimator
                quantized = x + (quantized - x).detach()

                result_dict.update({
                    "quantizer_loss": loss,
                    "commitment_loss": commitment_loss,
                    "codebook_loss": codebook_loss,
                    "entropy_loss": entropy_loss,
                })

            result_dict["encodings"] = encodings
            result_dict["encoding_indices"] = encoding_indices

        return quantized.to(x.dtype), result_dict

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Quantize the input encodings using the codebook.

        Args:
            encodings (torch.Tensor): One-hot encodings of the input.

        Returns:
            torch.Tensor: Quantized vectors.
        """
        return torch.matmul(encodings, self.codebook)

    def get_codebook(self) -> torch.Tensor:
        """
        Get the current codebook.

        Returns:
            torch.Tensor: The codebook.
        """
        return self.codebook

    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Decode quantization indices to their corresponding vectors.

        Args:
            ids (torch.Tensor): Tensor of quantization indices.

        Returns:
            torch.Tensor: Decoded vectors.
        """
        return F.embedding(ids, self.codebook)


class LookupFreeQuantizer(Quantizer):
    """
    Lookup-free Vector Quantizer implementation.
    """

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        entropy_loss_ratio: float = 0.0,
        entropy_loss_type: str = "softmax",
        entropy_temperature: float = 1.0,
        latent_normalize: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.entropy_loss_ratio = entropy_loss_ratio
        self.entropy_loss_type = entropy_loss_type
        self.entropy_temperature = entropy_temperature
        self.latent_normalize = latent_normalize

        assert (
            codebook_size & (codebook_size - 1) == 0
        ), "Codebook size must be a power of 2"
        self.codebook_dim = int(math.log2(codebook_size))
        self.register_buffer("basis", 2 ** torch.arange(self.codebook_dim - 1, -1, -1))
        self.register_buffer(
            "codebook",
            self.indices_to_codes(torch.arange(self.codebook_size, dtype=torch.long)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        flat_x = x.view(-1, self.embedding_dim)

        with torch.autocast(enabled=False):
            if self.latent_normalize:
                flat_x = l2_normalize(flat_x, dim=-1)
                codebook = l2_normalize(self.codebook.float(), dim=-1)

            distances = squared_euclidean_distance(flat_x, codebook)
            quantized = self.quantize(flat_x)
            encoding_indices = self.codes_to_indices(quantized)

            result_dict = dict()
            if self.training:
                commitment_loss = F.mse_loss(x, quantized.detach())

                if self.entropy_loss_ratio != 0:
                    entropy_loss = (
                        entropy_loss(
                            -distances.view(x.shape[:-1] + (-1,)),
                            loss_type=self.entropy_loss_type,
                            temperature=self.entropy_temperature,
                        )
                        * self.entropy_loss_ratio
                    )

                loss = commitment_loss * self.commitment_cost + entropy_loss

                # Straight-through estimator
                quantized = x + (quantized - x).detach()

                result_dict.update({
                    "quantizer_loss": loss,
                    "commitment_loss": commitment_loss,
                    "entropy_loss": entropy_loss,
                })

            result_dict["encoding_indices"] = encoding_indices

        return quantized.to(x.dtype), result_dict

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Quantizes the input encodings using the sign function.

        Args:
            encodings (torch.Tensor): Input tensor to be quantized.

        Returns:
            torch.Tensor: Quantized tensor with values -1 or 1.
        """
        return torch.sign(encodings)

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Converts binary codes to their corresponding indices.

        Args:
            codes (torch.Tensor): Binary codes tensor with values -1 or 1.

        Returns:
            torch.Tensor: Tensor of indices corresponding to the input codes.
        """
        return ((codes > 0).int() * self.basis.int()).sum(dim=-1)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Converts indices to their corresponding binary codes.

        Args:
            indices (torch.Tensor): Tensor of indices to be converted.

        Returns:
            torch.Tensor: Binary codes tensor with values -1 or 1.
        """
        binary = (indices[:, None] & self.basis != 0).float()
        return binary * 2 - 1
