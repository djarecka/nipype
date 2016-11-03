# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..preprocess import BlurToFWHM


def test_BlurToFWHM_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    automask=dict(argstr='-automask',
    ),
    blurmaster=dict(argstr='-blurmaster %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    fwhm=dict(argstr='-FWHM %f',
    ),
    fwhmxy=dict(argstr='-FWHMxy %f',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-input %s',
    mandatory=True,
    ),
    mask=dict(argstr='-blurmaster %s',
    ),
    out_file=dict(argstr='-prefix %s',
    name_source=[u'in_file'],
    name_template='%s_afni',
    ),
    outputtype=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = BlurToFWHM.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_BlurToFWHM_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = BlurToFWHM.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
