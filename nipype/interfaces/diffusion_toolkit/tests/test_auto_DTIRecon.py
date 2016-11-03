# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..dti import DTIRecon


def test_DTIRecon_inputs():
    input_map = dict(DWI=dict(argstr='%s',
    mandatory=True,
    position=1,
    ),
    args=dict(argstr='%s',
    ),
    b0_threshold=dict(argstr='-b0_th',
    ),
    bvals=dict(mandatory=True,
    ),
    bvecs=dict(argstr='-gm %s',
    mandatory=True,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    image_orientation_vectors=dict(argstr='-iop %f',
    ),
    n_averages=dict(argstr='-nex %s',
    ),
    oblique_correction=dict(argstr='-oc',
    ),
    out_prefix=dict(argstr='%s',
    position=2,
    usedefault=True,
    ),
    output_type=dict(argstr='-ot %s',
    usedefault=True,
    ),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = DTIRecon.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_DTIRecon_outputs():
    output_map = dict(ADC=dict(),
    B0=dict(),
    FA=dict(),
    FA_color=dict(),
    L1=dict(),
    L2=dict(),
    L3=dict(),
    V1=dict(),
    V2=dict(),
    V3=dict(),
    exp=dict(),
    tensor=dict(),
    )
    outputs = DTIRecon.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
