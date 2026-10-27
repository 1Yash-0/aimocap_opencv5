"""OpenCap LabValidation evaluation harness.

The package turns one trial of the OpenCap LabValidation archive into a
calibrated, synchronised, triangulated and scored run of the production
``aimocap`` pipeline.  Modules:

``dataset``      extract one trial from the archive into a self-describing folder
``calibration``  checkerboard extrinsics solved with the shipped intrinsics
``crops``        person-centred crop videos and the mapping back to source pixels
``sync``         visibility-aware geometric camera synchronisation and window cut
``truth``        mocap-derived per-camera offsets (evaluation only)
``mocap``        TRC parsing, virtual joints and rigid alignment helpers
``scoring``      accuracy and smoothness metrics against marker mocap
``render``       2D overlay grid and 3D skeleton-versus-mocap videos
``events``       JSON Lines stage log
``run_trial``    command-line entry point that runs every stage

Only ``truth`` and ``scoring`` read marker mocap.  Reconstruction stages
(2D inference, synchronisation, triangulation) never see it.
"""
