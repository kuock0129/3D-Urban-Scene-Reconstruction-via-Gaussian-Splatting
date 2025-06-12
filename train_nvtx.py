#!/usr/bin/env python3

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
# from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import numpy as np

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim_loss
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from utils.general_utils import safe_state
import lpipsPyTorch

# Import NVTX for NSIGHT analysis
from torch.cuda import nvtx

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, 
             checkpoint, debug_from, data_type="kitti360", affine=False):
    
    # NVTX: Mark training initialization
    nvtx.range_push("Training Setup")

    # Initialize Gaussian models
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    
    # Create Gaussian models
    gaussians = GaussianModel(dataset.sh_degree, affine=affine)
    scene = Scene(dataset, gaussians, data_type=data_type)
    gaussians.training_setup(opt)
    
    # Initialize background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Load checkpoint if provided
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
        # Load dynamic Gaussians
        for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
            dynamic_checkpoint = checkpoint.replace("chkpnt", f"dynamic_{iid}_chkpnt")
            if os.path.exists(dynamic_checkpoint):
                (dynamic_params, _) = torch.load(dynamic_checkpoint)
                dynamic_gaussian.restore(dynamic_params, opt)
        
        # Load unicycle models
        for iid, unicycle_pkg in scene.unicycles.items():
            unicycle_checkpoint = checkpoint.replace("chkpnt", f"unicycle_{iid}_chkpnt")
            if os.path.exists(unicycle_checkpoint):
                unicycle_params = torch.load(unicycle_checkpoint)
                unicycle_pkg['model'].restore(unicycle_params)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
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
        nvtx.range_pop()  # End Learning Rate Update
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            nvtx.range_push("SH Degree Update")
            gaussians.oneupSHdegree()
            nvtx.range_pop()  # End SH Degree Update

        # Pick a random Camera
        # NVTX: Camera selection
        nvtx.range_push("Camera Selection")
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Get previous camera for optical flow if available
        train_cameras = scene.getTrainCameras()
        prev_cam = None
        if iteration > 1000:  # Start optical flow training after warmup
            for i, cam in enumerate(train_cameras):
                if cam.uid == viewpoint_cam.uid and i > 0:
                    prev_cam = train_cameras[i-1]
                    break
        nvtx.range_pop()  # End Camera Selection

        # Render
        # NVTX: Rendering pass
        nvtx.range_push("Forward Pass - Rendering")
        render_pkg = render(viewpoint_cam, prev_cam, gaussians, scene.dynamic_gaussians, 
                          scene.unicycles, pipe, background, render_optical=(prev_cam is not None))
        
        image, feats, depth, opticalflow, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], render_pkg["feats"], render_pkg["depth"], 
            render_pkg["opticalflow"], render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )
        nvtx.range_pop()  # End Forward Pass - Rendering

        # Loss computation
        # NVTX: Loss computation
        nvtx.range_push("Loss Calculation")

        gt_image = viewpoint_cam.original_image.cuda()
        
        # RGB Loss
        Ll1 = l1_loss(image, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_loss(image, gt_image))
        
        # Semantic Loss
        loss_semantic = 0.0
        if viewpoint_cam.semantic2d is not None and iteration > 500:
            gt_semantic = viewpoint_cam.semantic2d.cuda()
            semantic_probs = F.softmax(feats, dim=0)
            loss_semantic = F.cross_entropy(feats.unsqueeze(0), gt_semantic) * 0.1
        
        # Optical Flow Loss
        loss_optical = 0.0
        if prev_cam is not None and viewpoint_cam.optical_gt is not None and iteration > 2000:
            gt_optical = viewpoint_cam.optical_gt.cuda()
            loss_optical = F.mse_loss(opticalflow[:2], gt_optical.permute(2, 0, 1)[:2]) * 0.01
        
        # Dynamic regularization loss
        loss_dynamic_reg = 0.0
        for track_id, unicycle_pkg in scene.unicycles.items():
            loss_dynamic_reg += unicycle_pkg['model'].reg_loss() * 0.001
            loss_dynamic_reg += unicycle_pkg['model'].pos_loss() * 0.01
        
        # Total loss
        loss = loss_rgb + loss_semantic + loss_optical + loss_dynamic_reg
        nvtx.range_pop()  # End Loss Calculation

        # NVTX: Backward pass
        nvtx.range_push("Backward Pass")
        loss.backward()
        nvtx.range_pop()  # End Backward Pass

        iter_end.record()

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
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                        #   testing_iterations, scene, render, (pipe, background), data_type)
            
            if iteration in saving_iterations:
                nvtx.range_push("Model Saving")
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)
                nvtx.range_pop()  # End Model Saving

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                nvtx.range_push("Optimization Step")
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                # Update dynamic Gaussians
                for dynamic_gaussian in scene.dynamic_gaussians.values():
                    if hasattr(dynamic_gaussian, 'optimizer'):
                        dynamic_gaussian.optimizer.step()
                        dynamic_gaussian.optimizer.zero_grad(set_to_none=True)
                nvtx.range_pop()  # End Optimization Step
                
                # Update unicycle models
                for unicycle_pkg in scene.unicycles.values():
                    unicycle_pkg['optimizer'].step()
                    unicycle_pkg['optimizer'].zero_grad(set_to_none=True)
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

    # Create Tensorboard logger
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, data_type):
    # NVTX: Training report
    nvtx.range_push("Training Report")
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and train stats
    if iteration in testing_iterations:
        nvtx.range_push("Validation")
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                                         for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                nvtx.range_push(f"Validation - {config['name']}")
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    nvtx.range_push(f"Validation Render {idx}")
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
                    nvtx.range_pop()  # End Validation Render
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                nvtx.range_pop()  # End Validation config

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        nvtx.range_pop()  # End Validation
    nvtx.range_pop()  # End Training Report


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
             args.start_checkpoint, args.debug_from, args.data_type, args.affine)

    # All done
    print("\nTraining complete.")