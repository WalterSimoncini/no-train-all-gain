"""
    This test suite should be run on the CPU, as non-deterministic
    operations on the GPU may yield different gradients depending
    on the batch size

    https://stackoverflow.com/questions/61292150/breaking-down-a-batch-in-pytorch-leads-to-different-results-why
    https://stackoverflow.com/questions/49609226/pytorch-purpose-of-addmm-function
"""
import torch
import torch.nn as nn
import src.utils.autograd_hacks as autograd_hacks

from collections import OrderedDict
from src.utils.grads import get_layer
from torch.nn.functional import softmax
from src.utils.models import freeze_model
from src.utils.transforms import Patchify
from grads_simclr import step as simclr_step
from grads_kl import step as kl_grads_step
from grads_dino import DINOHead, DINODataAgumentation, step as dino_step
from .fixtures import (
    dino,
    device,
    scaler,
    images_batch,
    sample_images,
    simclr_comparison_batch,
    dino_gradients_layer_path
)


def test_batched_dino_grads(dino, dino_gradients_layer_path, sample_images, device, scaler):
    """
        This test verifies that the per-sample gradients
        resulting from the DINO loss do not change with
        respect to the batch size
    """
    student = DINOHead(backbone=dino, embeddings_dim=768, output_dim=768).to(device)
    teacher = DINOHead(
        backbone=torch.hub.load("facebookresearch/dino:main", "dino_vitb16"),
        embeddings_dim=768,
        output_dim=768
    ).to(device)

    transform = DINODataAgumentation()
    dino_gradients_layer_path = f"backbone.{dino_gradients_layer_path}"

    # Freeze all layers except the target layers
    freeze_model(model=teacher)
    freeze_model(model=student, exclusions=[dino_gradients_layer_path])

    # Add hooks to calculate the per-sample gradients
    autograd_hacks.add_hooks(student, layer_paths=[dino_gradients_layer_path])

    images_batch = torch.cat(
        [transform(x).unsqueeze(dim=0) for x in sample_images],
        dim=0
    ).to(device)

    loss = dino_step(
        student=student,
        teacher=teacher,
        views=images_batch[0].unsqueeze(dim=0),
        device=device,
        use_fp16=True,
        scaler=scaler
    )

    autograd_hacks.compute_grad1(student, layer_paths=[dino_gradients_layer_path])

    assert not torch.isnan(loss)

    # Get the layer we calculate gradients for
    layer = get_layer(model=student, path=dino_gradients_layer_path)

    # Validate that gradients were stored and have the correct shape
    assert layer.weight.grad is not None
    assert layer.weight.grad1 is not None

    # We use 12 views for the DINO loss
    assert layer.weight.grad1.shape == torch.Size([12, 768, 768])

    single_sample_grad = layer.weight.grad1.detach().clone()
    single_sample_grad = single_sample_grad.reshape(1, 12, 768, 768)
    single_sample_grad = single_sample_grad.sum(dim=1)

    # Two samples (batched) forward/backward
    loss = dino_step(
        student=student,
        teacher=teacher,
        views=images_batch,
        device=device,
        use_fp16=True,
        scaler=scaler
    )

    autograd_hacks.compute_grad1(student, layer_paths=[dino_gradients_layer_path])

    assert not torch.isnan(loss)

    # Validate that gradients were stored and have the correct shape
    assert layer.weight.grad is not None
    assert layer.weight.grad1 is not None
    # We now have double the amount of views
    assert layer.weight.grad1.shape == torch.Size([24, 768, 768])

    two_sample_grad = layer.weight.grad1.reshape(2, 12, 768, 768)
    two_sample_grad = two_sample_grad.sum(dim=1)

    # It's impossible to get the exact same gradients, but we are happy
    # if they are close enough to the single-item batch gradients
    assert torch.isclose(
        two_sample_grad[0], single_sample_grad, atol=10e-5
    ).sum() >= (768 * 768) * 0.99


def test_batched_simclr_grads(dino, dino_gradients_layer_path, sample_images, simclr_comparison_batch, device, scaler):
    """
        This test verifies that the per-sample gradients
        resulting from the SimCLR loss do not change with
        respect to the batch size
    """
    # Wrap the model in a sequential to make it behave
    # as a SimCLR-style model
    dino = nn.Sequential(OrderedDict([
        ("feature_extractor", nn.Sequential(
            dino,
            nn.Linear(in_features=768, out_features=96)
        )),
    ])).to(device)

    # We add the feature_extractor prefix as we
    # wrap our model with a sequential block
    dino_gradients_layer_path = f"feature_extractor.0.{dino_gradients_layer_path}"

    # Freeze all layers except the target layer
    freeze_model(model=dino, exclusions=[dino_gradients_layer_path])

    # Add hooks to calculate the per-sample gradients
    autograd_hacks.add_hooks(dino, layer_paths=[dino_gradients_layer_path])

    # Load the patchify transform
    transform = Patchify(num_patches=4)
    images_batch = torch.cat(
        [transform(x).unsqueeze(dim=0) for x in sample_images],
        dim=0
    ).to(device)

    loss = simclr_step(
        model=dino,
        test_views=images_batch[0].unsqueeze(dim=0),
        comparison_batch=simclr_comparison_batch,
        device=device,
        use_fp16=True,
        scaler=scaler
    )

    autograd_hacks.compute_grad1(dino, layer_paths=[dino_gradients_layer_path])

    assert not torch.isnan(loss)

    # Get the layer we calculate gradients for
    layer = get_layer(model=dino, path=dino_gradients_layer_path)

    # Validate that gradients were stored and have the correct shape
    assert layer.weight.grad is not None
    assert layer.weight.grad1 is not None
    # We use a patch that has half of the width of the image
    # and a "stride" that is 1/4 the patch size, so we expect
    # to have 25 patches.
    assert layer.weight.grad1.shape == torch.Size([25, 768, 768])

    single_sample_grad = layer.weight.grad1.detach().clone()
    single_sample_grad = single_sample_grad.reshape(1, 25, 768, 768)
    single_sample_grad = single_sample_grad.sum(dim=1)

    # Two samples (batched) forward/backward
    loss = simclr_step(
        model=dino,
        test_views=images_batch,
        comparison_batch=simclr_comparison_batch,
        device=device,
        use_fp16=False,
        scaler=scaler
    )

    autograd_hacks.compute_grad1(dino, layer_paths=[dino_gradients_layer_path])

    assert not torch.isnan(loss)

    # Validate that gradients were stored and have the correct shape
    assert layer.weight.grad is not None
    assert layer.weight.grad1 is not None
    assert layer.weight.grad1.shape == torch.Size([50, 768, 768])

    two_sample_grad = layer.weight.grad1.reshape(2, 25, 768, 768)
    two_sample_grad = two_sample_grad.sum(dim=1)

    # It's impossible to get the exact same gradients, but we are happy
    # if they are close enough to the single-item batch gradients
    assert torch.isclose(
        two_sample_grad[0], single_sample_grad, atol=10e-5
    ).sum() >= (768 * 768) * 0.99


def test_batched_kl_grads(dino, dino_gradients_layer_path, images_batch, device, scaler):
    """
        This test verifies that the per-sample gradients
        resulting from the KL loss do not change with
        respect to the batch size
    """
    batch_size, temperature, embedding_size = 2, 15, 768

    # Freeze all layers except the target layer
    freeze_model(model=dino, exclusions=[dino_gradients_layer_path])

    # Add hooks to calculate the per-sample gradients
    autograd_hacks.add_hooks(dino, layer_paths=[dino_gradients_layer_path])

    uniform = (torch.ones(embedding_size) / embedding_size).to(device)

    softmax_uniform = softmax(uniform / temperature, dim=0)
    softmax_uniform = softmax_uniform.unsqueeze(dim=0).repeat(batch_size, 1)

    # Single sample forward/backward
    loss = kl_grads_step(
        model=dino,
        images=images_batch[0].unsqueeze(dim=0),
        device=device,
        softmax_uniform=softmax_uniform[0].unsqueeze(dim=0),
        use_fp16=True,
        scaler=scaler
    )

    autograd_hacks.compute_grad1(dino, layer_paths=[dino_gradients_layer_path])

    assert not torch.isnan(loss)

    # Get the layer we calculate gradients for
    layer = get_layer(model=dino, path=dino_gradients_layer_path)

    # Validate that gradients were stored and have the correct shape
    assert layer.weight.grad is not None
    assert layer.weight.grad1 is not None
    assert layer.weight.grad1.shape == torch.Size([1, 768, 768])

    single_sample_grad = layer.weight.grad1.detach().clone()

    # Two samples (batched) forward/backward
    loss = kl_grads_step(
        model=dino,
        images=images_batch,
        device=device,
        softmax_uniform=softmax_uniform,
        use_fp16=True,
        scaler=scaler
    )

    autograd_hacks.compute_grad1(dino, layer_paths=[dino_gradients_layer_path])

    assert not torch.isnan(loss)

    # Validate that gradients were stored and have the correct shape
    assert layer.weight.grad is not None
    assert layer.weight.grad1 is not None
    assert layer.weight.grad1.shape == torch.Size([2, 768, 768])

    # It's impossible to get the exact same gradients, but we are happy
    # if they are close enough to the single-item batch gradients
    assert torch.isclose(
        layer.weight.grad1[0, :], single_sample_grad, atol=10e-5
    ).sum() >= (768 * 768) * 0.99
