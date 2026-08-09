"""Remove fMRI gradient (imaging) artifacts from EEG recorded during MRI."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.pick import _picks_to_idx
from ..utils import _check_option, _validate_type, logger, verbose


class GradientRemover:
    """Remove the fMRI gradient artifact using robust average templates.

    Implements a volume-locked variant of the average artifact subtraction
    (AAS) method of :footcite:`AllenEtAl2000`. For each imaging volume (TR), an
    artifact template is formed from nearby volumes and subtracted. Volumes
    with excessive MRI-estimated head motion can be excluded from template
    construction, followed by rejection of volumes whose artifact waveform
    correlates poorly with the evolving template.

    This class operates on a plain :class:`~numpy.ndarray` and exposes the
    intermediate templates and rejection diagnostics. Most users should
    prefer :func:`mne.preprocessing.remove_fmri_gradient_artifact`, which
    operates directly on a :class:`~mne.io.Raw` object.

    Parameters
    ----------
    eeg_data : ndarray, shape (n_channels, n_times)
        The raw EEG data to perform gradient correction on.
    tr_events : ndarray
        The sample numbers at which imaging volumes begin. May be a 1D array
        of sample numbers, shape ``(n_trs,)``, or an ``(n_trs, 3)`` events
        array as returned by :func:`mne.find_events` (the first column is
        used). TRs must be evenly spaced in time, within ``tr_tol`` samples.
    n_average : int
        The requested number of nearby, motion-eligible volumes used to
        construct each template. The target volume is not included. If too
        few volumes pass correlation rejection, fewer volumes are used.
        Default 25.
    n_seed : int
        The number of motion-eligible volumes initially averaged without
        correlation rejection. Default 5.
    correlation_threshold : float | None
        The minimum median channel-wise correlation between a candidate
        volume and the evolving template. If ``None``, do not perform
        correlation rejection. Default 0.975.
    motion : path-like | ndarray | None
        MRI-estimated motion parameters, with one row per ``tr_events`` entry
        and six columns. A path is read using :func:`numpy.loadtxt`. The
        column order and units are determined by ``motion_source``.
    motion_source : ``'afni'`` | ``'fsl'`` | ``'spm'`` | None
        Software that produced ``motion``. AFNI parameters must be ordered
        ``roll, pitch, yaw, dS, dL, dP`` with rotations in degrees and
        translations in mm. FSL parameters must contain X/Y/Z rotations in
        radians followed by X/Y/Z translations in mm. SPM parameters must
        contain X/Y/Z translations in mm followed by pitch/roll/yaw rotations
        in radians.
    motion_threshold : float | None
        Maximum motion score in mm for a volume to contribute to an artifact
        template. If ``None``, all volumes are motion eligible. Default
        ``None``.
    motion_metric : ``'framewise_displacement'`` | ``'euclidean_norm'``
        Metric used to calculate the motion score from consecutive volumes.
        ``'framewise_displacement'`` sums the absolute translation and
        rotation changes, while ``'euclidean_norm'`` calculates their
        Euclidean norm. Rotations are converted to displacement using
        ``head_radius`` before either metric is calculated. Default
        ``'framewise_displacement'``.
    head_radius : float
        Head radius in mm used to convert rotation changes to displacement.
        Default 50.
    tr_tol : int
        The maximum allowed deviation in samples of an individual TR spacing
        from the median spacing. Default 0.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        eeg_data,
        tr_events,
        n_average=25,
        n_seed=5,
        correlation_threshold=0.975,
        motion=None,
        motion_source=None,
        motion_threshold=None,
        motion_metric="framewise_displacement",
        head_radius=50.0,
        tr_tol=0,
    ):
        _validate_type(eeg_data, np.ndarray, "eeg_data")
        _validate_type(tr_events, np.ndarray, "tr_events")
        if eeg_data.ndim != 2:
            raise ValueError(
                "eeg_data must have shape (n_channels, n_times), "
                f"but received shape {eeg_data.shape}."
            )
        self._n_average = _ensure_positive_int(n_average, "n_average")
        self._n_seed = _ensure_positive_int(n_seed, "n_seed")
        if self.n_seed > self.n_average:
            raise ValueError(
                f"n_seed ({self.n_seed}) must not exceed n_average ({self.n_average})."
            )
        self._correlation_threshold = _ensure_correlation_threshold(
            correlation_threshold
        )
        self._tr_events = self._valid_tr_events(tr_events, tr_tol)
        if self._tr_events[-1] + self.tr_spacing > eeg_data.shape[1]:
            raise ValueError(
                f"Last TR event is sample {self._tr_events[-1]} but "
                f"eeg data only contains {eeg_data.shape[1]} samples. "
                "Please check your tr event markers."
            )
        self._data = eeg_data
        self._motion_parameters = _load_motion(motion, motion_source, n_tr=self.n_tr)
        _check_option(
            "motion_metric",
            motion_metric,
            ("framewise_displacement", "euclidean_norm"),
        )
        self._motion_metric = motion_metric
        self._motion_score = _compute_motion_score(
            self.motion_parameters, head_radius, motion_threshold, motion_metric
        )
        if motion_threshold is None:
            self._motion_eligible = np.ones(self.n_tr, bool)
        else:
            self._motion_eligible = self.motion_score <= float(motion_threshold)
        if self.motion_eligible.sum() < self.n_seed + 1:
            raise ValueError(
                f"At least {self.n_seed + 1} motion-eligible volumes are required "
                f"but only {self.motion_eligible.sum()} were available."
            )
        self._candidate_indices = [None] * self.n_tr
        self._correlations = [None] * self.n_tr
        self._correlation_eligible = [None] * self.n_tr
        self._template_indices = [None] * self.n_tr
        self._templates = [None] * self.n_tr
        self._corrected = None

    @property
    def corrected(self):
        """The gradient-corrected data (computed on first access)."""
        if self._corrected is not None:
            return self._corrected
        return self.correct()

    @property
    def n_average(self):
        """The requested number of volumes used for each template."""
        return self._n_average

    @property
    def n_seed(self):
        """The number of volumes used to seed each template."""
        return self._n_seed

    @property
    def correlation_threshold(self):
        """The minimum correlation for inclusion in a template."""
        return self._correlation_threshold

    @property
    def tr_spacing(self):
        """The median number of samples between consecutive TRs."""
        return int(np.round(np.median(np.diff(self._tr_events))))

    @property
    def n_tr(self):
        """The number of imaging volumes."""
        return len(self._tr_events)

    @property
    def n_channels(self):
        """The number of channels."""
        return len(self._data)

    @property
    def motion_parameters(self):
        """Motion parameters normalized to translations in mm and rotations in rad."""
        return self._motion_parameters

    @property
    def motion_metric(self):
        """The metric used to calculate the motion score."""
        return self._motion_metric

    @property
    def motion_score(self):
        """The motion score in mm, or ``None`` if motion was omitted."""
        return self._motion_score

    @property
    def motion_eligible(self):
        """Boolean mask of volumes eligible according to head motion."""
        return self._motion_eligible

    @property
    def candidate_indices(self):
        """Candidate volume indices considered for each template."""
        return tuple(self._candidate_indices)

    @property
    def correlations(self):
        """Candidate correlations for each template."""
        return tuple(self._correlations)

    @property
    def correlation_eligible(self):
        """Correlation eligibility masks for each template."""
        return tuple(self._correlation_eligible)

    @property
    def template_indices(self):
        """Accepted volume indices used for each template."""
        return tuple(self._template_indices)

    def get_tr(self, n):
        """Get the uncorrected data at a given TR.

        Parameters
        ----------
        n : int
            The TR to get the uncorrected data at (0-indexed).

        Returns
        -------
        data : ndarray, shape (n_channels, tr_spacing)
            The uncorrected data at the given TR.
        """
        this_start, this_end = self._tr_bounds(n)
        return self._data[:, this_start:this_end]

    def get_tr_template(self, n):
        """Get the gradient artifact template at a given TR.

        Parameters
        ----------
        n : int
            The TR to get the template at (0-indexed).

        Returns
        -------
        template : ndarray, shape (n_channels, tr_spacing)
            The artifact template at the given TR.
        """
        self._check_valid_tr(n)
        if self._templates[n] is not None:
            return self._templates[n]
        candidates = self._get_candidate_indices(n)
        accepted = list(candidates[: self.n_seed])
        examined = list(accepted)
        correlations = [np.nan] * self.n_seed
        correlation_eligible = [True] * self.n_seed
        template = np.mean([self.get_tr(tr) for tr in accepted], axis=0)
        for tr in candidates[self.n_seed :]:
            if len(accepted) == self.n_average:
                break
            correlation = _median_channel_correlation(self.get_tr(tr), template)
            examined.append(tr)
            correlations.append(correlation)
            if (
                self.correlation_threshold is not None
                and correlation < self.correlation_threshold
            ):
                correlation_eligible.append(False)
                continue
            correlation_eligible.append(True)
            accepted.append(tr)
            template += (self.get_tr(tr) - template) / len(accepted)
        self._candidate_indices[n] = np.asarray(examined, int)
        self._correlations[n] = np.asarray(correlations)
        self._correlation_eligible[n] = np.asarray(correlation_eligible, bool)
        self._template_indices[n] = np.asarray(accepted, int)
        self._templates[n] = template
        return template

    def get_tr_corrected(self, n):
        """Get the gradient-corrected data at a given TR.

        Parameters
        ----------
        n : int
            The TR to get the corrected data at (0-indexed).

        Returns
        -------
        data : ndarray, shape (n_channels, tr_spacing)
            The gradient-corrected data at the given TR.
        """
        return self.get_tr(n) - self.get_tr_template(n)

    def correct(self):
        """Generate the gradient-corrected data.

        Returns
        -------
        corrected : ndarray, shape (n_channels, n_times)
            The gradient-corrected data.
        """
        corrected = self._data.copy()
        for tr in range(self.n_tr):
            this_start, this_end = self._tr_bounds(tr)
            corrected[:, this_start:this_end] = self.get_tr_corrected(tr)
        self._corrected = corrected
        return corrected

    def _get_candidate_indices(self, n):
        eligible = np.flatnonzero(self.motion_eligible)
        eligible = eligible[eligible != n]
        order = np.lexsort((eligible, np.abs(eligible - n)))
        return eligible[order]

    @staticmethod
    def _valid_tr_events(tr_events, tr_tol=0):
        if tr_events.ndim == 2:
            if tr_events.shape[1] == 3:
                tr_events = tr_events[:, 0]
            else:
                raise ValueError(
                    "TRs must be a 1D array or a (N, 3) ndarray from mne. "
                    f"Received array of shape {tr_events.shape}."
                )
        elif tr_events.ndim != 1:
            raise ValueError(
                "TRs must be a 1D array or a (N, 3) ndarray from mne. "
                f"Received array of shape {tr_events.shape}."
            )
        if len(tr_events) < 2:
            raise ValueError("At least two TR events are required.")
        if not np.issubdtype(tr_events.dtype, np.number):
            raise TypeError("TR events must contain numeric sample numbers.")
        if not np.isfinite(tr_events).all():
            raise ValueError("TR events must contain only finite sample numbers.")
        if not np.array_equal(tr_events, np.round(tr_events)):
            raise ValueError("TR events must contain integer sample numbers.")
        tr_events = np.asarray(np.round(tr_events), int)
        tr_tol = _ensure_nonnegative_number(tr_tol, "tr_tol")
        if tr_tol != int(tr_tol):
            raise TypeError(f"tr_tol must be an integer (received {tr_tol}).")
        diffs = np.diff(tr_events)
        if np.any(diffs <= 0):
            raise ValueError("TR events must be strictly increasing.")
        median_spacing = np.median(diffs)
        deviation = np.abs(diffs - median_spacing)
        if np.any(deviation > tr_tol):
            unique = np.unique(diffs)
            raise ValueError(
                "TR spacings are not consistent (median spacing "
                f"{median_spacing}, tolerance {tr_tol} samples); the "
                f"following unique distances were present: {unique}."
            )
        return tr_events

    def _check_valid_tr(self, n):
        if n < 0 or n >= self.n_tr:
            raise ValueError(f"Index {n} not in TR range [0, {self.n_tr - 1}]")

    def _tr_bounds(self, n):
        self._check_valid_tr(n)
        this_start = self._tr_events[n]
        this_end = this_start + self.tr_spacing
        return (this_start, this_end)


def _ensure_positive_int(value, name):
    _validate_type(value, "int-like", name)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer (received {value}).")
    return int(value)


def _ensure_nonnegative_number(value, name):
    _validate_type(value, "numeric", name)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite (received {value}).")
    if value < 0:
        raise ValueError(f"{name} must be non-negative (received {value}).")
    return value


def _ensure_correlation_threshold(value):
    if value is None:
        return None
    value = float(_ensure_nonnegative_number(value, "correlation_threshold"))
    if value > 1:
        raise ValueError(
            "correlation_threshold must be between 0 and 1, inclusive "
            f"(received {value})."
        )
    return value


def _load_motion(motion, motion_source, *, n_tr):
    if motion is None:
        if motion_source is not None:
            raise ValueError("motion_source cannot be provided when motion is None.")
        return None
    if motion_source is None:
        raise ValueError("motion_source must be provided when motion is provided.")
    _check_option("motion_source", motion_source, ("afni", "fsl", "spm"))
    _validate_type(motion, (np.ndarray, "path-like"), "motion")
    if isinstance(motion, np.ndarray):
        parameters = motion
    else:
        parameters = np.loadtxt(motion, ndmin=2)
    if parameters.ndim != 2 or parameters.shape[1] != 6:
        raise ValueError(
            f"motion must have shape (n_trs, 6), but received shape {parameters.shape}."
        )
    if len(parameters) != n_tr:
        raise ValueError(
            "motion must contain exactly one row per TR event, but received "
            f"{len(parameters)} motion rows and {n_tr} TR events."
        )
    if not np.issubdtype(parameters.dtype, np.number):
        raise TypeError("motion must contain numeric parameters.")
    parameters = np.asarray(parameters, float)
    if not np.isfinite(parameters).all():
        raise ValueError("motion must contain only finite parameters.")
    if motion_source == "afni":
        rotations = np.deg2rad(parameters[:, [1, 2, 0]])
        translations = parameters[:, [4, 5, 3]]
    elif motion_source == "fsl":
        rotations = parameters[:, :3]
        translations = parameters[:, 3:]
    else:  # spm
        translations = parameters[:, :3]
        rotations = parameters[:, 3:]
    return np.concatenate([translations, rotations], axis=1)


def _compute_motion_score(parameters, head_radius, motion_threshold, motion_metric):
    head_radius = float(_ensure_nonnegative_number(head_radius, "head_radius"))
    if motion_threshold is not None:
        _ensure_nonnegative_number(motion_threshold, "motion_threshold")
        if parameters is None:
            raise ValueError("motion must be provided when motion_threshold is set.")
    if parameters is None:
        return None
    differences = np.diff(parameters, axis=0)
    differences[:, 3:] *= head_radius
    if motion_metric == "framewise_displacement":
        score = np.abs(differences).sum(axis=1)
    else:
        score = np.linalg.norm(differences, axis=1)
    return np.concatenate([[0.0], score])


def _median_channel_correlation(epoch, template):
    epoch = epoch - epoch.mean(axis=1, keepdims=True)
    template = template - template.mean(axis=1, keepdims=True)
    numerator = np.sum(epoch * template, axis=1)
    denominator = np.linalg.norm(epoch, axis=1) * np.linalg.norm(template, axis=1)
    correlations = np.ones(len(epoch))
    nonzero = denominator > np.finfo(float).eps
    correlations[nonzero] = numerator[nonzero] / denominator[nonzero]
    correlations[~nonzero & np.any(epoch != template, axis=1)] = 0.0
    return np.median(correlations)


@verbose
def remove_fmri_gradient_artifact(
    raw,
    tr_events,
    *,
    n_average=25,
    n_seed=5,
    correlation_threshold=0.975,
    motion=None,
    motion_source=None,
    motion_threshold=None,
    motion_metric="framewise_displacement",
    head_radius=50.0,
    tr_tol=0,
    picks=None,
    copy=True,
    verbose=None,
):
    """Remove the fMRI gradient (imaging) artifact from EEG data.

    Removes the gradient artifact present in EEG recorded simultaneously with
    functional MRI using robust, volume-locked average artifact subtraction
    (AAS) :footcite:`AllenEtAl2000`. For each imaging volume, nearby volumes
    are screened using MRI-estimated motion and correlation with the evolving
    artifact template before averaging and subtraction.

    See :ref:`tut-fmri-gradient` for a full example.

    Parameters
    ----------
    raw : instance of Raw
        The raw data recorded during MRI acquisition. Must be preloaded.
    tr_events : ndarray
        The sample numbers at which imaging volumes begin. May be a 1D array
        of sample numbers, shape ``(n_trs,)``, or an ``(n_trs, 3)`` events
        array as returned by :func:`mne.find_events` (the first column is
        used). TRs must be evenly spaced in time, within ``tr_tol`` samples.
    n_average : int
        The requested number of nearby, motion-eligible volumes used to
        construct each template. The target volume is not included. If too
        few volumes pass correlation rejection, fewer volumes are used.
        Default 25.
    n_seed : int
        The number of motion-eligible volumes initially averaged without
        correlation rejection. Default 5.
    correlation_threshold : float | None
        The minimum median channel-wise correlation between a candidate
        volume and the evolving template. If ``None``, do not perform
        correlation rejection. Default 0.975.
    motion : path-like | ndarray | None
        MRI-estimated motion parameters, with one row per ``tr_events`` entry
        and six columns. A path is read using :func:`numpy.loadtxt`.
    motion_source : ``'afni'`` | ``'fsl'`` | ``'spm'`` | None
        Software that produced ``motion``. See :class:`GradientRemover` for
        the expected column order and units for each source.
    motion_threshold : float | None
        Maximum motion score in mm for a volume to contribute to a template.
        If ``None``, all volumes are motion eligible. Default ``None``.
    motion_metric : ``'framewise_displacement'`` | ``'euclidean_norm'``
        Metric used to calculate the motion score from consecutive volumes.
        See :class:`GradientRemover` for details. Default
        ``'framewise_displacement'``.
    head_radius : float
        Head radius in mm used to convert rotation changes to displacement.
        Default 50.
    tr_tol : int
        The maximum allowed deviation in samples of an individual TR spacing
        from the median spacing. Default 0.
    %(picks_all_data_noref)s
    copy : bool
        If True (default), operate on and return a copy of ``raw``. If False,
        modify ``raw`` in place.
    %(verbose)s

    Returns
    -------
    raw : instance of Raw
        The raw data with the gradient artifact removed.

    Notes
    -----
    .. versionadded:: 1.13

    This implementation uses one epoch per imaging volume. Slice-level
    interpolation and adaptive noise cancellation from the complete method of
    :footcite:`AllenEtAl2000` require slice timing information and are not
    performed.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(copy, bool, "copy")
    _validate_type(tr_events, np.ndarray, "tr_events")

    if not raw.preload:
        raise RuntimeError(
            "raw data must be preloaded to remove the gradient artifact, use "
            "raw.load_data() or preload=True when reading the data."
        )

    picks = _picks_to_idx(raw.info, picks, none="data", exclude="bads")

    if copy:
        raw = raw.copy()

    data = raw.get_data(picks=picks)
    remover = GradientRemover(
        data,
        tr_events,
        n_average=n_average,
        n_seed=n_seed,
        correlation_threshold=correlation_threshold,
        motion=motion,
        motion_source=motion_source,
        motion_threshold=motion_threshold,
        motion_metric=motion_metric,
        head_radius=head_radius,
        tr_tol=tr_tol,
    )
    raw._data[picks] = remover.correct()
    if motion_threshold is not None:
        n_motion_bad = np.sum(~remover.motion_eligible)
        logger.info(
            f"Excluded {n_motion_bad} of {remover.n_tr} volumes from template "
            f"construction based on MRI-estimated motion ({motion_metric})"
        )
    if correlation_threshold is not None:
        n_candidates = sum(len(indices) for indices in remover.candidate_indices)
        n_accepted = sum(len(indices) for indices in remover.template_indices)
        logger.info(
            f"Excluded {n_candidates - n_accepted} of {n_candidates} candidate "
            "volumes based on artifact correlation"
        )
    template_sizes = np.array([len(indices) for indices in remover.template_indices])
    logger.info(
        f"Templates contained {template_sizes.min()}-{template_sizes.max()} volumes "
        f"(median {np.median(template_sizes):g})"
    )

    return raw
