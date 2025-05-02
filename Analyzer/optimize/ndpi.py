import os
import subprocess
from pathlib import Path
from typing import Dict, Union
from tifffile import TiffFile

def get_meta_info_ndpi(ndpi_file: Path) -> Dict[str, Union[int, float]]:
    """
    Extract meta information from an ndpi file.

    Args:
        ndpi_file: Path to the ndpi file.

    Returns:
        Dictionary with width, height (both in the in x5 coordinate system) and shape ratio of the image file.
    """
    tiff = TiffFile(ndpi_file)
    if len(tiff.pages) == 0:
        tiff.close()
        raise ValueError(
            f'The file {ndpi_file} does not contain any tiff pages. Cannot read tiff data (maybe the file is broken?). Aborting...')

    page = tiff.pages[0]  # x20
    width = page.shape[1] / 4
    height = page.shape[0] / 4
    shape_ratio = width / height
    tiff.close()

    return {
        'width': width,
        'height': height,
        'shape_ratio': shape_ratio
    }

def convert_ndpi_to_tiff(ndpi_file: Path, target_file: Path, crop_region: str, scale: str) -> None:
    """
    Takes a crop of an ndpi file and converts it to a tiff file.

    Args:
        ndpi_file: Path to the ndpi file.
        target_file: Path to the target tiff file.
        crop_region: The region to take from the ndpi file. Specified in [0,1] coordinates as a string with 4 values 'x,y,width,height' (x and y specify the top left corner in the image). See the -e option of ndpisplit for more infos (https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/).
        scale: The scaling of the target file (e.g. x5 or x20).
    """

    assert ndpi_file.is_file(), 'No ndpi file given as input'
    assert target_file.suffix == '.tif', 'Can only convert to tif'
    assert scale.startswith('x'), 'Scale should be specified as x5, x20 etc.'
    
    subprocess.run(
        f'ndpisplit -e{crop_region},LABEL -{scale} -c0j -O {target_file.parent} {str(ndpi_file)}',
        shell=True)
    ndpisplit_result_file = target_file.parent / f'{ndpi_file.stem}_{scale}_z0_LABEL.tif'

    # Sometimes ndpisplit fails without giving any warnings. The best we can do is to check if the result file exists
    if not ndpisplit_result_file.exists():
        raise ValueError(
            f'ndpisplit failed to convert the file {ndpi_file} to tiff (maybe the file is broken?). Aborting...')

    # Rename the file from the default filename of ndpisplit to the target filename
    os.rename(ndpisplit_result_file, target_file)

