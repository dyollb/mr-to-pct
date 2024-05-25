import copy
import os

import ants
import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import NibabelWriter
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    HistogramNormalize,
    LoadImage,
    ResizeWithPadOrCrop,
    ScaleIntensity,
    ToTensor,
)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .netdef import ShuffleUNet


# define patch-based inference function
# patches are 256 x 256 x 32 (i.e. 32 slices in axial/transverse plane)
def do_inference_3D(
    net, x, device, dims=(256, 256), patch_size=(256, 256, 32), step_size=2
):
    # pad image
    y = np.ones_like(x)
    pad_val = x[5, 5, 5]  # find background value
    y *= 0
    in_shape = x.shape[0:2]
    z_size = patch_size[2]
    pad_dims = tuple(
        ((d1 - d2) // 2, (d1 - d2) - (d1 - d2) // 2) for d1, d2 in zip(dims, in_shape)
    )
    pad_dims = pad_dims + ((0, z_size * 2),)
    x_pad = np.pad(x, pad_dims, "constant", constant_values=pad_val)
    y_pad = np.pad(y, pad_dims, "constant", constant_values=0)

    num_overlaps = np.zeros(y_pad.shape)

    # iterate through patches & infer
    zz = 0
    while zz + z_size < x_pad.shape[2]:
        xp = x_pad[np.newaxis, np.newaxis, :, :, zz : zz + z_size]
        txp = ToTensor()(xp).to(device)
        output = net(txp)
        output = output[0, 0, ...].cpu().data.numpy()
        y_pad[:, :, zz : zz + z_size] += output
        num_overlaps[:, :, zz : zz + z_size] += 1
        zz += step_size

    y = y_pad[
        pad_dims[0][0] : y_pad.shape[0] - pad_dims[0][1],
        pad_dims[1][0] : y_pad.shape[1] - pad_dims[1][1],
        pad_dims[2][0] : y_pad.shape[2] - pad_dims[2][1],
    ]
    num_overlaps = num_overlaps[
        pad_dims[0][0] : y_pad.shape[0] - pad_dims[0][1],
        pad_dims[1][0] : y_pad.shape[1] - pad_dims[1][1],
        pad_dims[2][0] : y_pad.shape[2] - pad_dims[2][1],
    ]
    y = y / num_overlaps

    return y


# Prepare T1w MRI: bias correction and create head mask
def do_prep_t1(img_file):
    img = ants.image_read(img_file)
    img_n4 = ants.n4_bias_field_correction(img)
    img_tmp = img_n4.otsu_segmentation(k=3)  # otsu_segmentation
    img_tmp = ants.multi_label_morphology(img_tmp, "MD", 2)  # dilate 2
    img_tmp = ants.smooth_image(img_tmp, 3)  # smooth 3
    img_tmp = ants.threshold_image(img_tmp, 0.5)  # threshold 0.5
    img_mask = ants.get_mask(img_tmp)
    img_out = ants.multiply_images(img_n4, img_mask)
    ants.image_write(img_out, (os.path.splitext(img_file)[0] + "_prep.nii"))
    print(
        "Prepare MR image done, output saved to: {}".format(
            os.path.splitext(img_file)[0] + "_prep.nii"
        )
    )


# load MRIs
def do_mr_to_pct(
    input_mr_file, output_pct_file, saved_model, device, prep_t1, plot_mrct
):
    # set network parameters and check net shape
    net = ShuffleUNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=2,
    )
    # n = torch.rand(1, 1, 256, 256, 32)
    # print(net(n).shape)  # should be [1, 1, 256, 256, 32]

    # specify transforms
    spatial_size = [256, 256, 256]
    t1trans_test = Compose(
        [
            HistogramNormalize(num_bins=256, min=0, max=255),
            ScaleIntensity(minv=-1.0, maxv=1.0),
            EnsureChannelFirst(),
            ResizeWithPadOrCrop(spatial_size, mode="minimum"),
        ]
    )

    # load images
    print(f"Loading MR image: {input_mr_file}")
    if prep_t1:
        print("Preparing MR image: bias correction and masking...")
        do_prep_t1(input_mr_file)
        t1_arr, t1_meta = LoadImage(image_only=False)(
            os.path.splitext(input_mr_file)[0] + "_prep.nii"
        )
    else:
        t1_arr, t1_meta = LoadImage(image_only=False)(input_mr_file)
    orig_t1_arr = copy.copy(t1_arr)
    t1_arr = t1trans_test(t1_arr)
    t1_arr = t1_arr[0]

    net.to(device)
    net.load_state_dict(saved_model)
    net.eval()

    print("Running MR to pCT...")
    with torch.no_grad():
        ct_out = do_inference_3D(net, t1_arr, device)

    ct_out = ResizeWithPadOrCrop(t1_meta["spatial_shape"], mode="minimum")(
        ct_out[np.newaxis, :]
    )
    writer = NibabelWriter()
    writer.set_data_array(ct_out[0, ...], channel_dim=None)
    writer.set_metadata({"affine": orig_t1_arr.affine})
    writer.write(output_pct_file, verbose=True)
    print(f"MR to pCT done, output saved to: {output_pct_file}")
    print("")
    print("Please inspect your output.")

    if plot_mrct:
        print("")
        print("Plotting an example slice at z = 150")
        # plot an example slice at z = 150
        z = 150
        fig = plt.figure(figsize=(10, 10))
        # plot input MR
        ax1 = fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(orig_t1_arr[:, :, z], cmap="gray")
        ax1.set_title(f"T1w MR input at z = {z}")
        ax1.axis("off")
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax1)

        # plot output pCT
        ax2 = fig.add_subplot(1, 2, 2)
        im2 = ax2.imshow(ct_out[0, :, :, z], cmap="gray")
        ax2.set_title(f"pCT output at z = {z}")
        ax2.axis("off")
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax2)

        plt.tight_layout()
