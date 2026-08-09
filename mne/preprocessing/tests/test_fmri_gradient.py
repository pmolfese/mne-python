# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne import create_info
from mne.io import RawArray
from mne.preprocessing import GradientRemover, remove_fmri_gradient_artifact
from mne.utils import catch_logging

N_TRS = 30
TR_CODE = 1
SAMPS_PER_TR = 100
N_CHANNELS = 8


def _sample_trs():
    return np.arange(N_TRS) * SAMPS_PER_TR


def _sample_trs_longform():
    return np.c_[_sample_trs(), np.zeros(N_TRS, int), np.full(N_TRS, TR_CODE)]


def _sample_data():
    return np.zeros((N_CHANNELS, N_TRS * SAMPS_PER_TR))


def _repeating_data():
    artifact = np.random.default_rng(42).standard_normal((N_CHANNELS, SAMPS_PER_TR))
    return np.tile(artifact, (1, N_TRS))


def test_parameter_validity():
    """Test validation of template construction parameters."""
    data, trs = _sample_data(), _sample_trs()
    with pytest.raises(ValueError, match="eeg_data must have shape"):
        GradientRemover(data[0], trs)
    for name, kwargs in (
        ("n_average", dict(n_average=0)),
        ("n_seed", dict(n_seed=0)),
    ):
        with pytest.raises(ValueError, match=rf"{name} must be a positive"):
            GradientRemover(data, trs, **kwargs)
    with pytest.raises(TypeError, match="n_average must be an instance"):
        GradientRemover(data, trs, n_average=2.5)
    with pytest.raises(ValueError, match="must not exceed n_average"):
        GradientRemover(data, trs, n_average=4, n_seed=5)
    for threshold in (-0.1, 1.1):
        with pytest.raises(ValueError, match="correlation_threshold"):
            GradientRemover(data, trs, correlation_threshold=threshold)
    with pytest.raises(ValueError, match="correlation_threshold must be finite"):
        GradientRemover(data, trs, correlation_threshold=np.nan)

    remover = GradientRemover(data, trs, n_average=10, n_seed=3)
    assert remover.n_average == 10
    assert remover.n_seed == 3
    assert remover.correlation_threshold == 0.975
    remover = GradientRemover(data, trs, n_average=5, n_seed=5)
    assert len(remover.get_tr_template(0)) == N_CHANNELS
    assert len(remover.template_indices[0]) == 5


def test_tr_events_validity():
    """Test validation of tr_events."""
    data = _sample_data()
    with pytest.raises(ValueError, match="At least two TR"):
        GradientRemover(data, np.array([0]))
    with pytest.raises(ValueError, match=r"TRs must be a 1D array or"):
        GradientRemover(data, np.array([[1, 2], [1, 2]]))
    with pytest.raises(ValueError, match="integer sample numbers"):
        GradientRemover(data, _sample_trs().astype(float) + 0.1)
    with pytest.raises(ValueError, match="strictly increasing"):
        GradientRemover(data, _sample_trs()[::-1])
    with pytest.raises(ValueError, match="tr_tol must be non-negative"):
        GradientRemover(data, _sample_trs(), tr_tol=-1)

    trs = _sample_trs()
    trs[1] += 5
    with pytest.raises(ValueError, match="TR spacings are not consistent"):
        GradientRemover(data, trs)
    with pytest.raises(ValueError, match="TR spacings are not consistent"):
        GradientRemover(data, np.c_[trs, np.zeros((N_TRS, 2), int)])

    remover = GradientRemover(data, _sample_trs_longform())
    assert remover.tr_spacing == SAMPS_PER_TR
    assert remover.n_tr == N_TRS


def test_tr_events_jitter_tolerance():
    """Test that small TR-spacing jitter is tolerated but not accumulated."""
    trs = _sample_trs().copy()
    trs[5] += 1
    remover = GradientRemover(_sample_data(), trs, tr_tol=1)
    assert remover.tr_spacing == SAMPS_PER_TR
    assert remover._tr_bounds(5)[0] == trs[5]

    trs[5] += 1
    with pytest.raises(ValueError, match="TR spacings are not consistent"):
        GradientRemover(_sample_data(), trs, tr_tol=1)


def test_template_construction_and_edges():
    """Test moving leave-one-out template construction at all volumes."""
    data = _repeating_data()
    remover = GradientRemover(
        data, _sample_trs(), n_average=25, correlation_threshold=None
    )
    corrected = remover.correct()
    assert_allclose(corrected, 0, atol=1e-12)
    assert remover.corrected is corrected
    for target in (0, N_TRS // 2, N_TRS - 1):
        indices = remover.template_indices[target]
        assert len(indices) == 25
        assert target not in indices
        assert_allclose(remover.get_tr_template(target), data[:, :SAMPS_PER_TR])

    with pytest.raises(ValueError, match="Index -1"):
        remover.get_tr(-1)
    with pytest.raises(ValueError, match="Index"):
        remover.get_tr(N_TRS)


def test_correlation_rejection():
    """Test rejection and replacement of an atypical artifact volume."""
    data = _repeating_data()
    data[:, 10 * SAMPS_PER_TR : 11 * SAMPS_PER_TR] *= -1
    remover = GradientRemover(data, _sample_trs(), n_average=20)
    remover.get_tr_template(0)
    candidates = remover.candidate_indices[0]
    bad_idx = np.where(candidates == 10)[0].item()
    assert remover.correlations[0][bad_idx] < 0
    assert not remover.correlation_eligible[0][bad_idx]
    assert 10 not in remover.template_indices[0]
    assert len(remover.template_indices[0]) == 20

    remover = GradientRemover(
        data, _sample_trs(), n_average=20, correlation_threshold=None
    )
    remover.get_tr_template(0)
    assert 10 in remover.template_indices[0]


@pytest.mark.parametrize("motion_source", ("afni", "fsl", "spm"))
def test_motion_sources(motion_source, tmp_path):
    """Test normalization of AFNI, FSL, and SPM motion parameters."""
    translations = np.zeros((N_TRS, 3))
    rotations = np.zeros((N_TRS, 3))
    translations[1] = (1.0, 2.0, 3.0)
    rotations[1] = (0.01, 0.02, 0.03)
    expected = np.c_[translations, rotations]
    if motion_source == "afni":
        motion = np.c_[
            np.rad2deg(rotations[:, 2]),
            np.rad2deg(rotations[:, 0]),
            np.rad2deg(rotations[:, 1]),
            translations[:, 2],
            translations[:, 0],
            translations[:, 1],
        ]
    elif motion_source == "fsl":
        motion = np.c_[rotations, translations]
    else:
        motion = expected
    motion_path = tmp_path / f"motion_{motion_source}.txt"
    np.savetxt(motion_path, motion)
    remover = GradientRemover(
        _sample_data(),
        _sample_trs(),
        motion=motion_path,
        motion_source=motion_source,
        motion_threshold=8.0,
    )
    assert_allclose(remover.motion_parameters, expected)
    assert_allclose(remover.framewise_displacement[:3], (0.0, 9.0, 9.0))
    assert_array_equal(remover.motion_eligible[:3], np.array([True, False, False]))


def test_motion_validation():
    """Test validation of motion inputs."""
    data, trs = _sample_data(), _sample_trs()
    motion = np.zeros((N_TRS, 6))
    with pytest.raises(ValueError, match="must be provided when motion"):
        GradientRemover(data, trs, motion=motion)
    with pytest.raises(ValueError, match="cannot be provided"):
        GradientRemover(data, trs, motion_source="afni")
    with pytest.raises(ValueError, match="Invalid value for.*motion_source"):
        GradientRemover(data, trs, motion=motion, motion_source="bad")
    with pytest.raises(ValueError, match=r"shape \(n_trs, 6\)"):
        GradientRemover(data, trs, motion=motion[:, :5], motion_source="spm")
    with pytest.raises(ValueError, match="exactly one row per TR"):
        GradientRemover(data, trs, motion=motion[:-1], motion_source="spm")
    motion[0, 0] = np.nan
    with pytest.raises(ValueError, match="only finite"):
        GradientRemover(data, trs, motion=motion, motion_source="spm")
    with pytest.raises(ValueError, match="motion must be provided"):
        GradientRemover(data, trs, motion_threshold=0.5)


def test_motion_eligibility_and_diagnostics():
    """Test that high-motion volumes do not contribute to templates."""
    motion = np.zeros((N_TRS, 6))
    motion[10:, 0] = 1.0
    remover = GradientRemover(
        _repeating_data(),
        _sample_trs(),
        motion=motion,
        motion_source="spm",
        motion_threshold=0.5,
        n_average=20,
    )
    assert not remover.motion_eligible[10]
    remover.correct()
    for indices in remover.template_indices:
        assert 10 not in indices
        assert len(indices) == 20
    assert all(indices is not None for indices in remover.candidate_indices)
    assert all(mask is not None for mask in remover.correlation_eligible)

    motion[:, 0] = np.arange(N_TRS)
    with pytest.raises(ValueError, match="motion-eligible volumes are required"):
        GradientRemover(
            _sample_data(),
            _sample_trs(),
            motion=motion,
            motion_source="spm",
            motion_threshold=0.5,
        )


def test_remove_fmri_gradient_artifact():
    """Test the Raw-level wrapper."""
    info = create_info(N_CHANNELS, sfreq=100.0, ch_types="eeg")
    data = _repeating_data()
    raw = RawArray(data, info)
    motion = np.zeros((N_TRS, 6))
    motion[10:, 0] = 1.0

    with catch_logging() as log:
        out = remove_fmri_gradient_artifact(
            raw,
            _sample_trs(),
            motion=motion,
            motion_source="spm",
            motion_threshold=0.5,
            verbose=True,
        )
    assert out is not raw
    assert_allclose(raw.get_data(), data)
    assert_allclose(out.get_data(), 0, atol=1e-12)
    log = log.getvalue()
    assert "Excluded 1 of 30 volumes" in log
    assert "Templates contained 25-25 volumes" in log

    out = remove_fmri_gradient_artifact(raw, _sample_trs_longform(), copy=False)
    assert out is raw

    raw_nopreload = RawArray(data, info)
    raw_nopreload.preload = False
    with pytest.raises(RuntimeError, match="must be preloaded"):
        remove_fmri_gradient_artifact(raw_nopreload, _sample_trs())
