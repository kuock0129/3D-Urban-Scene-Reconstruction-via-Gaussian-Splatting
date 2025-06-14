 #
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_scale_and_shift, ScaleAndShiftInvariantLoss
from utils.general_utils import vis_depth, read_propagted_depth
from gaussian_renderer import network_gui
from gaussian_renderer import render2D as render

from utils.graphics_utils import surface_normal_from_depth, img_warping, depth_propagation, check_geometric_consistency, generate_edge_mask
import sys
from scene import Scene
from scene import GaussianModel2D as GaussianModel
from utils.general_utils import safe_state, load_pairs_relation
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio
import numpy as np
import torchvision
import cv2
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Define profiler activities to monitor
activities = [
ProfilerActivity.CPU,
ProfilerActivity.CUDA, # For GPU operations
]
# Configure profiling only for a specific epoch (e.g., the second epoch)
PROFILE_EPOCH = 0 # 0-indexed epoch
WARMUP_STEPS = 10 # Skip first few batches to avoid initialization overhead
ACTIVE_STEPS = 10 # Number of steps to actively profile
WAIT_STEPS = 5 # Additional steps to wait before warmup begins
# Create profiler schedule
prof_schedule = torch.profiler.schedule(
wait=WAIT_STEPS, # Skip these steps before warmup
warmup=WARMUP_STEPS, # Warmup steps (prepare profiler)
active=ACTIVE_STEPS, # Steps to actively profile
repeat=1 # No repeats needed for this use case
)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    #read the overlapping txt
    if opt.dataset == '360' and opt.depth_loss:
        pairs = load_pairs_relation(opt.pair_path)
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # instantiate the profiler with desired properties/config
    profiler = torch.profiler.profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/profile_epoch_{PROFILE_EPOCH}'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True)
    profiler.start()

    # depth_loss_fn = ScaleAndShiftInvariantLoss(alpha=0.1, scales=1)
    propagated_iteration_begin = opt.propagated_iteration_begin
    propagated_iteration_after = opt.propagated_iteration_after
    after_propagated = False
    propagation_dict = {}
    for i in range(0, len(viewpoint_stack), 1):
        propagation_dict[viewpoint_stack[i].image_name] = False

    for iteration in range(first_iter, opt.iterations + 1):        
        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        randidx = randint(0, len(viewpoint_stack)-1)
        # if iteration > propagated_iteration_begin and iteration < propagated_iteration_after and after_propagated:
        #     randidx = propagated_view_index
        viewpoint_cam = viewpoint_stack[randidx]
        
        # Render
        with record_function("forward_pass"):
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            #render_pkg = render(viewpoint_cam, gaussians, pipe, bg, return_normal=args.normal_loss)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                                return_normal=opt.normal_loss, return_opacity=True, return_depth=opt.depth_loss or opt.depth2normal_loss)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            
        # opacity mask
        if iteration < opt.propagated_iteration_begin and opt.depth_loss:
            opacity_mask = render_pkg['render_opacity'] > 0.999
            if len(opacity_mask.shape) == 2:
                opacity_mask = opacity_mask.unsqueeze(0)
            opacity_mask = opacity_mask.repeat(3, 1, 1)
        else:
            opacity_mask = render_pkg['render_opacity'] > 0.0
            if len(opacity_mask.shape) == 2:
                opacity_mask = opacity_mask.unsqueeze(0)
            opacity_mask = opacity_mask.repeat(3, 1, 1)

        # Loss
        with record_function("loss_calculation"):
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1 = l1_loss(image[opacity_mask], gt_image[opacity_mask])
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image, mask=opacity_mask))

            # flatten loss
            if opt.flatten_loss:
                scales = gaussians.get_scaling
                min_scale, _ = torch.min(scales, dim=1)
                min_scale = torch.clamp(min_scale, 0, 30)
                flatten_loss = torch.abs(min_scale).mean()
                loss += opt.lambda_flatten * flatten_loss

            # opacity loss
            if opt.sparse_loss:
                opacity = gaussians.get_opacity
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[visibility_filter].mean()
                loss += opt.lambda_sparse * sparse_loss

            if opt.normal_loss:
                rendered_normal = render_pkg['render_normal']
                if viewpoint_cam.normal is not None:
                    normal_gt = viewpoint_cam.normal.cuda()
                    if viewpoint_cam.sky_mask is not None:
                        filter_mask = viewpoint_cam.sky_mask.to(normal_gt.device).to(torch.bool)
                        normal_gt[~(filter_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                    filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)

                    l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                    cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                    loss += opt.lambda_l1_normal * l1_normal + opt.lambda_cos_normal * cos_normal
        with record_function("backward_pass"):
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if not torch.isnan(loss):
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                with record_function("densification"):
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        print(f"Densitify with {gaussians.get_xyz.shape[0]} points")
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()

            # Optimizer step
            with record_function("optimizer_step"):
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                
               
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        # Step the profiler (important for the schedule to work)
        profiler.step()
    
    profiler.stop()  # on_trace_ready fires
    # Print summary of top 20 operations by CUDA time
    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # After profiler.stop(), add memory analysis
    print("\n=== Memory Usage Analysis ===")
    print(profiler.key_averages().table(
        sort_by="cuda_memory_usage", 
        row_limit=20,
        header="Memory Usage by Operation"
    ))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[100, 7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[100, 7000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
