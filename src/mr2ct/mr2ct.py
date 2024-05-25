from pathlib import Path

import torch
import typer

from .utils.infer_funcs import do_mr_to_pct


def mr2ct(
    input_mr_file: Path,
    output_pct_file: Path,
    model_file: Path = Path("pretrained_net_final_20220825.pth"),
):
    """Loads a t1w MR image, generates a pseudo-ct image and saves it to file"""

    # set device, use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set trained model to load
    saved_model = torch.load(model_file, map_location=device)

    # Do you want to prepare the t1 image? This will perform bias correction and create a head mask
    # yes = True, no = False. Output will be saved to _prep.nii
    prep_t1 = True

    # Do you want to produce an example plot? yes = True, no = False.
    plot_mrct = False

    # Run MR to pCT
    do_mr_to_pct(
        str(input_mr_file),
        str(output_pct_file),
        saved_model,
        device,
        prep_t1,
        plot_mrct,
    )


def main():
    typer.run(mr2ct)


if __name__ == "__main__":
    main()
