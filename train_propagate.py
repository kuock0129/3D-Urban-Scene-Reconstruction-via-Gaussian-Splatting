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
from utils.graphics_utils import surface_normal_from_depth, img_warping, depth_propagation, check_geometric_consistency, generate_edge_mask

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, 
             data_type="kitti360", affine=False):
    
    # Initialize Gaussian models
    first_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    
    # Create Gaussian models
    gaussians = GaussianModel(dataset.sh_degree, affine=affine)
    scene = Scene(dataset, gaussians, data_type="waymo", ignore_dynamic=False)

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
    
    propagation_dict = {}
    for i in range(0, len(viewpoint_stack), 1):
        propagation_dict[viewpoint_stack[i].image_name] = False

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)
        for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
            dynamic_gaussian.update_learning_rate(iteration)
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random Camera
        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack[rand_idx]



        intervals = [-2, -1, 1, 2]
        src_idxs = [rand_idx+itv for itv in intervals if ((itv + rand_idx > 0) and (itv + rand_idx < len(viewpoint_stack)))]

        #propagate the gaussians first
        with torch.no_grad():
           if iteration > opt.propagated_iteration_begin and iteration < opt.propagated_iteration_after and (iteration % opt.propagation_interval == 0 and not propagation_dict[viewpoint_cam.image_name]):
            # if opt.depth_loss and iteration > propagated_iteration_begin and iteration < propagated_iteration_after and (iteration % opt.propagation_interval == 0):
                propagation_dict[viewpoint_cam.image_name] = True

                render_pkg = render(viewpoint_cam, prev_cam, gaussians, scene.dynamic_gaussians, 
                          scene.unicycles, pipe, background)

                projected_depth = render_pkg["render_depth"]

                # get the opacity that less than the threshold, propagate depth in these region

                sky_mask = None

                # get the propagated depth
                propagated_depth, normal = depth_propagation(viewpoint_cam, projected_depth, viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)

                # cache the propagated_depth
                viewpoint_cam.depth = propagated_depth

                #transform normal to camera coordinate
                R_w2c = torch.tensor(viewpoint_cam.R.T).cuda().to(torch.float32)
                # R_w2c[:, 1:] *= -1
                normal = (R_w2c @ normal.view(-1, 3).permute(1, 0)).view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)                
                valid_mask = propagated_depth != 300

                # calculate the abs rel depth error of the propagated depth and rendered depth & render color error
                render_depth = render_pkg['render_depth']
                abs_rel_error = torch.abs(propagated_depth - render_depth) / propagated_depth
                abs_rel_error_threshold = opt.depth_error_max_threshold - (opt.depth_error_max_threshold - opt.depth_error_min_threshold) * (iteration - opt.propagated_iteration_begin) / (opt.propagated_iteration_after - opt.propagated_iteration_begin)
                # color error
                render_color = render_pkg['render']
    

                color_error = torch.abs(render_color - viewpoint_cam.original_image.cuda())
                color_error = color_error.mean(dim=0).squeeze()
                error_mask = (abs_rel_error > abs_rel_error_threshold)

                # # calculate the photometric consistency
                ref_K = viewpoint_cam.intrinsics.cuda()
                #c2w
                ref_pose = viewpoint_cam.world_view_transform.transpose(0, 1).inverse()
                
                # calculate the geometric consistency
                geometric_counts = None
                for idx, src_idx in enumerate(src_idxs):
                    src_viewpoint = viewpoint_stack[src_idx]
                    #c2w
                    src_pose = src_viewpoint.world_view_transform.transpose(0, 1).inverse()
                    src_K = src_viewpoint.intrinsics.cuda()

                    # if src_viewpoint.depth is None:
                    src_render_pkg = render_pkg = render(viewpoint_cam, prev_cam, gaussians, scene.dynamic_gaussians, 
                        scene.unicycles, pipe, background)
                    src_projected_depth = src_render_pkg['render_depth']
                
                #get the src_depth first
                    src_depth, src_normal = depth_propagation(src_viewpoint, src_projected_depth, viewpoint_stack, src_idxs, opt.dataset, opt.patch_size)
                    src_viewpoint.depth = src_depth
                    # else:
                    #     src_depth = src_viewpoint.depth
                        
                    mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff = check_geometric_consistency(propagated_depth.unsqueeze(0), ref_K.unsqueeze(0), 
                                                                                                                 ref_pose.unsqueeze(0), src_depth.unsqueeze(0), 
                                                                                                                 src_K.unsqueeze(0), src_pose.unsqueeze(0), thre1=2, thre2=0.01)
                    
                    if geometric_counts is None:
                        geometric_counts = mask.to(torch.uint8)
                    else:
                        geometric_counts += mask.to(torch.uint8)
                        
                cost = geometric_counts.squeeze()
                cost_mask = cost >= 2       
                
                normal[~(cost_mask.unsqueeze(0).repeat(3, 1, 1))] = -10
                viewpoint_cam.normal = normal
                
                propagated_mask = valid_mask & error_mask & cost_mask
                if sky_mask is not None:
                    propagated_mask = propagated_mask & sky_mask

                propagated_depth[~cost_mask] = 300 
                # propagated_mask = propagated_mask & edge_mask
                propagated_depth[~propagated_mask] = 300
  
                if propagated_mask.sum() > 100:
                    gaussians.densify_from_depth_propagation(viewpoint_cam, propagated_depth, propagated_mask.to(torch.bool), gt_image) 

        # Get previous camera for optical flow if available
        train_cameras = scene.getTrainCameras()
        prev_cam = None
        if iteration > 1000:  # Start optical flow training after warmup
            for i, cam in enumerate(train_cameras):
                if cam.uid == viewpoint_cam.uid and i > 0:
                    prev_cam = train_cameras[i-1]
                    break

        # Render
        render_pkg = render(viewpoint_cam, prev_cam, gaussians, scene.dynamic_gaussians, 
                          scene.unicycles, pipe, background)
        
        image, depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],  render_pkg["render_depth"], 
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], render_pkg["radii"]
        )

        viewspace_point_tensor.retain_grad()
    
        # Loss computation
        gt_image = viewpoint_cam.original_image.cuda()
        
        # RGB Loss
        Ll1 = l1_loss(image, gt_image)
        loss_rgb = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss(image, gt_image)
        
        # Total loss
        loss = loss_rgb 

        reg_loss = 0
        if opt.uc_opt_pos and (len(scene.unicycles) > 0) and (1000 < iteration) and (iteration < 15000):
            for track_id, unicycle_pkg in scene.unicycles.items():
                model = unicycle_pkg['model']
                reg_loss += 1e-3 * model.reg_loss() + 1e-4 * model.pos_loss()
            reg_loss = reg_loss / len(scene.unicycles)
            loss += reg_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
            #               testing_iterations, scene, render, (pipe, background), data_type)
            
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    dynamic_gaussian.optimizer.step()
                    dynamic_gaussian.optimizer.zero_grad(set_to_none = True)

                if dataset.unicycle and opt.uc_opt_pos and iteration > 1000:
                    for track_id, unicycle_pkg in scene.unicycles.items():
                        unicycle_optimizer = unicycle_pkg['optimizer']
                        unicycle_optimizer.step()
                        unicycle_optimizer.zero_grad(set_to_none = True)

            # Densification
            if iteration < opt.densify_until_iter:

                # gsplat
                grad = viewspace_point_tensor.grad.clone()
                # grad[..., 0] *= viewpoint_cam.image_width / 2.0
                # grad[..., 1] *= viewpoint_cam.image_height / 2.0
                # Keep track of max radii in image-space for pruning
                current_index = gaussians.get_xyz.shape[0]
                gaussians.max_radii2D[visibility_filter[:current_index]] = torch.max(gaussians.max_radii2D[visibility_filter[:current_index]], radii[:current_index][visibility_filter[:current_index]])
                gaussians.add_densification_stats_grad(grad[:current_index], visibility_filter[:current_index])
                last_index = current_index

                for iid in viewpoint_cam.dynamics.keys():
                    dynamic_gaussian = scene.dynamic_gaussians[iid]
                    current_index = last_index + dynamic_gaussian.get_xyz.shape[0]
                    visible_mask = visibility_filter[last_index:current_index]
                    dynamic_gaussian.max_radii2D[visible_mask] = torch.max(dynamic_gaussian.max_radii2D[visible_mask], radii[last_index:current_index][visible_mask])
                    dynamic_gaussian.add_densification_stats_grad(grad[last_index:current_index], visible_mask)
                    last_index = current_index

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    print(f"Densifying with {gaussians.get_xyz.shape[0]} points")
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                        dynamic_gaussian.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or \
                    (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                        dynamic_gaussian.reset_opacity()

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                save_path = dataset.save_path + f"chkpnt{iteration}.pth"
               
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save((gaussians.capture(), iteration), save_path)
                
                # Save dynamic Gaussians
                for iid, dynamic_gaussian in scene.dynamic_gaussians.items():
                    save_dynamic_path = dataset.save_path + f"dynamic_{iid}_chkpnt{iteration}.pth"
                    torch.save((dynamic_gaussian.capture(), iteration), save_dynamic_path)
                if dataset.unicycle and opt.uc_opt_pos:
                    for track_id, unicycle_pkg in scene.unicycles.items():
                        model = unicycle_pkg['model']
                        save_unicycle_path = dataset.save_path + f"unicycle_{track_id}_chkpnt{iteration}.pth"
                        torch.save(model.capture(), save_unicycle_path)
                        save_unicycle_path = dataset.save_path + f"unicycle_{track_id}_chkpnt{iteration}.png"
                        model.visualize(save_unicycle_path)

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

    # # Create Tensorboard logger
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer


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