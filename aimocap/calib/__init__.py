"""Camera calibration and physical floor registration."""

from aimocap.calib.floor import (
    CharucoBoardModel,
    FloorModelError,
    FloorRegistrationResult,
    FloorValidationProfile,
    HoldCapture,
    ObservationSet,
    CameraObservation,
    PlanarAmbiguityError,
    RegistrationGateError,
    compute_floor_transform_from_board,
    evaluate_floor_registration,
    load_floor_registration,
    require_grounded_floor,
    save_floor_registration,
    solve_board_pose_multiview,
)

__all__ = [
    "CharucoBoardModel",
    "FloorModelError",
    "FloorRegistrationResult",
    "FloorValidationProfile",
    "HoldCapture",
    "ObservationSet",
    "CameraObservation",
    "PlanarAmbiguityError",
    "RegistrationGateError",
    "compute_floor_transform_from_board",
    "evaluate_floor_registration",
    "load_floor_registration",
    "require_grounded_floor",
    "save_floor_registration",
    "solve_board_pose_multiview",
]
