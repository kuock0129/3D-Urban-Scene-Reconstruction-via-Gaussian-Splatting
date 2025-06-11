#!/usr/bin/env python3

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
from random import randint
# from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import numpy as np
import gc
import uuid
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim_loss
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state
import lpipsPyTorch

# Import NVTX for NSIGHT analysis
from torch.cuda import nvtx

# Add to Scene/Camera loading - downsample by 2x
def downsample_image(image, factor=2):
    h, w = image.shape[-2:]
    return F.interpolate(image.unsqueeze(0), size=(h//factor, w//factor), 
                        mode='bilinear', align_corners=False).squeeze(0)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, 
             data_type="waymo", affine=False):
    
    # NVTX: Mark training initialization
    nvtx.range_push("Training Setup")

    # Initialize Gaussian models
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    
    # Create Gaussian models
    gaussians = GaussianModel(dataset.sh_degree, affine=affine)
    scene = Scene(dataset, gaussians, unicycle=False, uc_fit_iter=0, data_type="waymo", ignore_dynamic=False)

    gaussians.training_setup(opt)
    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
        dynamic_gaussian.training_setup(opt)
    
    # Initialize background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    nvtx.range_pop()  # End Training Setup

    # NVTX: Mark main training loop
    nvtx.range_push("Training Loop")
    
    for iteration in range(first_iter, opt.iterations + 1):
        # NVTX: Mark each training iteration
        nvtx.range_push(f"Training Iteration {iteration}")

        iter_start.record()

        # Update learning rate
        # NVTX: Learning rate updates
        nvtx.range_push("Learning Rate Update")
        gaussians.update_learning_rate(iteration)
        for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
            dynamic_gaussian.update_learning_rate(iteration)
        nvtx.range_pop()  # End Learning Rate Update
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            nvtx.range_push("SH Degree Update")
            gaussians.oneupSHdegree()
            nvtx.range_pop()  # End SH Degree Update

        # Pick a random Camera
        # NVTX: Camera selection
        nvtx.range_push("Camera Selection")

        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack[rand_idx]

        # Get previous camera for optical flow if available
        train_cameras = scene.getTrainCameras()
        prev_cam = None
        if iteration > 1000:  # Start optical flow training after warmup
            for i, cam in enumerate(train_cameras):
                if cam.uid == viewpoint_cam.uid and i > 0:
                    prev_cam = train_cameras[i-1]
                    break
        nvtx.range_pop()  # End Camera Selection
        
        # ===== ADD DYNAMIC OBJECT CULLING HERE =====
        # Only process dynamic objects visible in current frame
        visible_dynamics = {}
        if hasattr(viewpoint_cam, 'dynamics') and viewpoint_cam.dynamics:
            for iid in viewpoint_cam.dynamics.keys():
                if iid in scene.dynamic_gaussians:
                    visible_dynamics[iid] = scene.dynamic_gaussians[iid]
        else:
            visible_dynamics = {}
        
        print(f"Frame {iteration}: Using {len(visible_dynamics)}/{len(scene.dynamic_gaussians)} dynamic objects")
        # ============================================

        # Render
        # NVTX: Rendering pass
        nvtx.range_push("Forward Pass - Rendering")

        render_pkg = render(viewpoint_cam, prev_cam, gaussians, scene.dynamic_gaussians, 
                          scene.unicycles, pipe, background)
        
        image, depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],  render_pkg["depth"], 
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        nvtx.range_pop()  # End Forward Pass - Rendering

        # ONLY retain grad if we're in densification phase
        if iteration < opt.densify_until_iter and iteration % 20 == 0:  # Every 20 iterations instead
            nvtx.range_push("Gradient Retention")
            viewspace_point_tensor.retain_grad()
            nvtx.range_pop()  # End Gradient Retention
    
        # Loss computation
        # NVTX: Loss computation
        nvtx.range_push("Loss Calculation")
        # FIX: Downsample both images to match sizes
        gt_image = downsample_image(viewpoint_cam.original_image.cuda(), factor=2)
        image_downsampled = downsample_image(image, factor=2)
        
        # RGB Loss
        # Ll1 = l1_loss(image, gt_image)
        # loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss(image, gt_image)
        
        # RGB Loss using downsampled images
        Ll1 = l1_loss(image_downsampled, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss(image_downsampled, gt_image)
        
        # Total loss
        loss = loss_rgb 
        nvtx.range_pop()  # End Loss Calculation


        # NVTX: Backward pass
        nvtx.range_push("Backward Pass")
        loss.backward()
        nvtx.range_pop()  # End Backward Pass


        iter_end.record()

        static_mask = viewpoint_cam.mask.flatten()
        
        with torch.no_grad():
            # NVTX: Progress reporting
            nvtx.range_push("Progress Reporting")
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            nvtx.range_pop()  # End Progress Reporting

            # Log and save
            if iteration in saving_iterations:
                nvtx.range_push("Model Saving")
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                nvtx.range_pop()  # End Model Saving

            # Optimizer step
            if iteration < opt.iterations:
                nvtx.range_push("Optimization Step")
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    dynamic_gaussian.optimizer.step()
                    dynamic_gaussian.optimizer.zero_grad(set_to_none=True)
                nvtx.range_pop()  # End Optimization Step

            # Densification
            if iteration < opt.densify_until_iter and iteration % 20 == 0:  # Only when gradients available
                nvtx.range_push("Densification")
                
                # Check if gradients are available
                if viewspace_point_tensor.grad is not None:
                    # Extract and process gradients
                    nvtx.range_push("Gradient Processing")
                    grad = viewspace_point_tensor.grad.clone()
                    grad[..., 0] *= viewpoint_cam.image_width / 2.0
                    grad[..., 1] *= viewpoint_cam.image_height / 2.0
                    nvtx.range_pop()  # End Gradient Processing
                        
                    # NVTX: Radii tracking
                    nvtx.range_push("Radii Tracking")
                    # Keep track of max radii in image-space for pruning
                    current_index = gaussians.get_xyz.shape[0]
                    gaussians.max_radii2D[visibility_filter[:current_index]] = torch.max(
                        gaussians.max_radii2D[visibility_filter[:current_index]], 
                        radii[:current_index][visibility_filter[:current_index]]
                    )
                    gaussians.add_densification_stats_grad(grad[:current_index], visibility_filter[:current_index])
                    last_index = current_index
                
                for iid in viewpoint_cam.dynamics.keys():
                    dynamic_gaussian = scene.dynamic_gaussians[iid]
                    current_index = last_index + dynamic_gaussian.get_xyz.shape[0]
                    visible_mask = visibility_filter[last_index:current_index]
                    dynamic_gaussian.max_radii2D[visible_mask] = torch.max(
                        dynamic_gaussian.max_radii2D[visible_mask], 
                        radii[last_index:current_index][visible_mask]
                    )
                    dynamic_gaussian.add_densification_stats_grad(grad[last_index:current_index], visible_mask)
                    last_index = current_index
                nvtx.range_pop()  # End Radii Tracking

                # Clean up gradient tensor immediately
                del grad

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    nvtx.range_push("Densify and Prune")
                    size_threshold = 10 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold* 1.5, 0.01, scene.cameras_extent, size_threshold)
                    nvtx.range_pop()  # End Densify and Prune

                if iteration % opt.opacity_reset_interval == 0 or \
                    (dataset.white_background and iteration == opt.densify_from_iter):
                    nvtx.range_push("Opacity Reset")
                    gaussians.reset_opacity()
                    nvtx.range_pop()  # End Opacity Reset
                
                nvtx.range_pop()  # End Densification

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), 
                          scene.model_path + f"/ckpts/chkpnt{iteration}.pth")
                
                # Save dynamic Gaussians
                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    torch.save((dynamic_gaussian.capture(), iteration), 
                              scene.model_path + f"/ckpts/dynamic_{iid}_chkpnt{iteration}.pth")
                
                # Save unicycle models
                for iid, unicycle_pkg in scene.unicycles.items():
                    torch.save(unicycle_pkg['model'].capture(), 
                              scene.model_path + f"/ckpts/unicycle_{iid}_chkpnt{iteration}.pth")

        # CRITICAL: Delete all large variables before next iteration
        del image, gt_image, loss, loss_rgb, Ll1, render_pkg
        del depth, viewspace_point_tensor, visibility_filter, radii
        
        # Aggressive memory cleanup every 50 iterations
        if iteration % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
        # Additional cleanup after densification
        if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        nvtx.range_pop()  # End Training Iteration
    nvtx.range_pop()  # End Training Loop




def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, data_type):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and train stats
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                                         for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, None, scene.gaussians, scene.dynamic_gaussians, 
                                          scene.unicycles, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), 
                                           image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), 
                                               gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    
                    # Clean up validation tensors immediately
                    del image, gt_image, render_pkg
                    
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--data_type', type=str, default='waymo')
    parser.add_argument('--affine', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = get_combined_args(parser)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
             args.data_type, args.affine)

    # All done
    print("\nTraining complete.")