# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from .....testing import assert_equal
from ..diffusion import DiffusionWeightedVolumeMasking


def test_DiffusionWeightedVolumeMasking_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    inputVolume=dict(argstr='%s',
    position=-4,
    ),
    otsuomegathreshold=dict(argstr='--otsuomegathreshold %f',
    ),
    outputBaseline=dict(argstr='%s',
    hash_files=False,
    position=-2,
    ),
    removeislands=dict(argstr='--removeislands ',
    ),
    terminal_output=dict(nohash=True,
    ),
    thresholdMask=dict(argstr='%s',
    hash_files=False,
    position=-1,
    ),
    )
    inputs = DiffusionWeightedVolumeMasking.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_DiffusionWeightedVolumeMasking_outputs():
    output_map = dict(outputBaseline=dict(position=-2,
    ),
    thresholdMask=dict(position=-1,
    ),
    )
    outputs = DiffusionWeightedVolumeMasking.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
