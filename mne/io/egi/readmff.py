# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

"""EGI MFF reader backed by mffpy."""

import datetime
import math
import os.path as op
from glob import glob

import numpy as np

from ..._fiff.constants import FIFF
from ..._fiff.meas_info import _empty_info, _ensure_meas_date_none_or_dt
from ..._fiff.utils import _create_chs, _mult_cal_one
from ...annotations import Annotations
from ...utils import _check_fname, _soft_import, logger, verbose, warn
from ..base import BaseRaw
from .egimff import (
    REFERENCE_NAMES,
    _add_pns_channel_info,
    _read_locs,
)
from .events import _combine_triggers, _triage_include_exclude


def _dt_to_sample(event_dt, start_dt, sfreq):
    """Convert an event datetime to a sample index exactly."""
    td = event_dt - start_dt
    # timedelta stores days/seconds/microseconds as Python ints
    total_us = (
        td.days * 86_400_000_000 + td.seconds * 1_000_000 + td.microseconds
    )
    sfreq_int = round(sfreq)
    g = math.gcd(sfreq_int, 1_000_000)
    return total_us * (sfreq_int // g) // (1_000_000 // g)


def _read_header_mffpy(input_fname):
    """Read EGI MFF header information using mffpy.

    Parameters
    ----------
    input_fname : path-like
        Path to the MFF directory.

    Returns
    -------
    egi_info : dict
        Dictionary containing header information needed to construct a raw
        object and perform on-demand reads.

    Notes
    -----
    This function replaces ``_read_header`` from ``egimff.py`` for the
    mffpy-backed reader. Timing is converted using integer arithmetic to
    preserve exact sample alignment.
    """
    mffpy = _soft_import("mffpy", "reading EGI MFF data")
    from mffpy import Reader, XML

    reader = Reader(input_fname)

    # Check for the public mffpy APIs this reader relies on.
    missing = []
    if not hasattr(reader, "block_sample_counts"):
        missing.append("Reader.block_sample_counts")
    if not hasattr(reader, "mff_flavor"):
        missing.append("Reader.mff_flavor")
    try:
        XML._parse_time_str("2009-04-01T10:33:02.332000000-05:00")
    except Exception:
        missing.append("9-digit XML timestamp parsing")
    if missing:
        missing = ", ".join(missing)
        raise ImportError(
            "The mffpy-backed raw MFF reader requires a newer mffpy with "
            f"public support for {missing}. Found mffpy {mffpy.__version__}. "
            "See https://pypi.org/project/mffpy/ or "
            "https://github.com/BEL-Public/mffpy."
        )
    flavor = reader.mff_flavor
    logger.info('    mffpy detected MFF flavor "%s"', flavor)
    if flavor != "continuous":
        raise ValueError(
            f"{input_fname} is a {flavor} MFF file. "
            "The mffpy-backed raw reader only supports continuous MFF files. "
            "Use mne.read_evokeds_mff() for segmented or averaged MFF data."
        )

    # --- Basic EEG signal info ---
    sfreq = reader.sampling_rates["EEG"]
    n_channels = reader.num_channels["EEG"]
    sfreq_int = round(sfreq)
    g = math.gcd(sfreq_int, 1_000_000)
    mult = sfreq_int // g
    div = 1_000_000 // g

    # --- Recording start time ---
    meas_dt_local = reader.startdatetime
    utc_offset = meas_dt_local.strftime("%z")

    # --- Device name and EEG channel list from sensorLayout.xml ---
    layout = XML.from_file(op.join(input_fname, "sensorLayout.xml"))
    device = layout.name
    numbers = []
    chan_type = []
    for _, sensor in sorted(layout.sensors.items()):
        if sensor["type"] in (0, 1):
            numbers.append(str(sensor["number"]).encode())
            chan_type.append("eeg")
    if len(numbers) != n_channels:
        raise RuntimeError(
            f"Number of defined channels ({len(numbers)}) did not match the "
            f"expected channels ({n_channels})."
        )

    # --- Epochs → first_samps, last_samps, eeg_sample_blocks, disk_samps ---
    epochs_xml = XML.from_file(op.join(input_fname, "epochs.xml"))
    epochs = epochs_xml.epochs
    first_samps_list = []
    last_samps_list = []
    samples_block_list = []
    for ep in epochs:
        f = int(ep.beginTime * mult // div)
        l = int(ep.endTime * mult // div)
        first_samps_list.append(f)
        last_samps_list.append(l)
        samples_block_list.append(l - f)
    first_samps = np.array(first_samps_list, dtype=np.int64)
    last_samps = np.array(last_samps_list, dtype=np.int64)
    eeg_sample_blocks = np.array(samples_block_list, dtype=np.int64)
    n_epochs = len(first_samps)
    has_gaps = bool(np.any(first_samps[1:] > last_samps[:-1])) if n_epochs > 1 else False
    eeg_block_counts = np.array(reader.block_sample_counts["EEG"], dtype=np.int64)
    logger.info(
        "    epochs.xml describes %d acquisition span%s (%s)",
        n_epochs,
        "" if n_epochs == 1 else "s",
        "with gaps" if has_gaps else "contiguous",
    )
    logger.info(
        "    EEG signal uses %d binary block%s",
        len(eeg_block_counts),
        "" if len(eeg_block_counts) == 1 else "s",
    )
    bad = (
        not (first_samps < last_samps).all()
        or not (first_samps[1:] >= last_samps[:-1]).all()
        or eeg_sample_blocks.sum() != eeg_block_counts.sum()
    )
    if not bad:
        for ep, n_epoch_samples in zip(epochs, eeg_sample_blocks):
            if eeg_block_counts[ep.block_slice].sum() != n_epoch_samples:
                bad = True
                break
    if bad:
        raise RuntimeError(
            "EGI epoch/block sample counts could not be reconciled:\n"
            f"first_samps={list(first_samps)}\n"
            f"last_samps={list(last_samps)}\n"
            f"eeg_block_counts={list(eeg_block_counts)}"
        )

    # Virtual timeline: slot i holds the physical disk position of sample i,
    # or -1 where there is an acquisition gap between epochs.
    disk_samps = np.full(int(last_samps[-1]), -1, dtype=np.int64)
    offset = 0
    for f, l in zip(first_samps, last_samps):
        n_this = int(l - f)
        disk_samps[f:l] = np.arange(offset, offset + n_this)
        offset += n_this

    # --- PNS channels ---
    pns_names = []
    pns_types = []
    pns_units = []
    pns_sample_blocks = None

    pns_xml_path = op.join(input_fname, "pnsSet.xml")
    if op.isfile(pns_xml_path) and "PNSData" in reader.sampling_rates:
        pns_set = XML.from_file(pns_xml_path)
        for _, sensor in sorted(pns_set.sensors.items()):
            name = sensor["name"]
            unit = sensor.get("unit", "")
            stype = sensor.get("sensorType", "").upper()
            if stype == "ECG":
                ch_type = "ecg"
            elif stype == "EMG":
                ch_type = "emg"
            else:
                ch_type = "bio"
            pns_names.append(name)
            pns_types.append(ch_type)
            pns_units.append(unit)

        # Detect the EGI PSG sample bug, where the last PNS block has one
        # fewer sample than the corresponding EEG block.
        pns_sample_blocks = np.array(
            reader.block_sample_counts.get("PNSData", eeg_sample_blocks),
            dtype=np.int64,
        )

    return dict(
        sfreq=sfreq,
        n_channels=n_channels,
        meas_dt_local=meas_dt_local,
        utc_offset=utc_offset,
        device=device,
        chan_type=chan_type,
        numbers=numbers,
        first_samps=first_samps,
        last_samps=last_samps,
        eeg_sample_blocks=eeg_sample_blocks,
        disk_samps=disk_samps,
        pns_names=pns_names,
        pns_types=pns_types,
        pns_units=pns_units,
        pns_sample_blocks=pns_sample_blocks,
        # Populated later by _read_mff_events_mffpy:
        n_events=0,
        event_codes=[],
    )


def _read_mff_events_mffpy(input_fname, egi_info):
    """Read MFF event tracks using mffpy.

    Parameters
    ----------
    input_fname : path-like
        Path to the MFF directory.
    egi_info : dict
        Header information dictionary returned by
        :func:`_read_header_mffpy`.

    Returns
    -------
    events : ndarray, shape (n_events, n_samples)
        Event matrix used internally for event-channel synthesis.
    egi_info : dict
        Updated header information dictionary.
    mff_events : dict
        Dictionary mapping event codes to lists of event sample indices.

    Notes
    -----
    This function replaces ``_read_events`` from ``events.py`` for the
    mffpy-backed reader.
    """
    from mffpy import Reader, XML

    reader = Reader(input_fname)
    start_dt = reader.startdatetime
    sfreq = egi_info["sfreq"]
    n_samples = egi_info["last_samps"][-1]

    codes = []
    markers = []  # list of (code, sample_int)

    for evfile in sorted(glob(op.join(input_fname, "Events_*.xml"))):
        track = XML.from_file(evfile)
        for event in track.events:
            code = event.get("code")
            if code is None:
                continue
            sample = _dt_to_sample(event["beginTime"], start_dt, sfreq)
            if code not in codes:
                codes.append(code)
            markers.append((code, sample))

    mff_events = {code: [] for code in codes}
    for code, sample in markers:
        mff_events[code].append(sample)

    egi_info["n_events"] = len(codes)
    egi_info["event_codes"] = codes

    events = np.zeros([len(codes), n_samples])
    for n, code in enumerate(codes):
        for i in mff_events[code]:
            if 0 <= i < n_samples:
                events[n, i] = n + 1

    return events, egi_info, mff_events


@verbose
def _read_raw_egi_mff_mffpy(
    input_fname,
    eog=None,
    misc=None,
    include=None,
    exclude=None,
    preload=False,
    channel_naming="E%d",
    *,
    events_as_annotations=True,
    verbose=None,
):
    """Read a continuous EGI MFF file using the mffpy backend.

    Parameters
    ----------
    input_fname : path-like
        Path to the MFF directory.
    eog : list | tuple | None
        Names of channels or list of indices that should be designated
        EOG channels.
    misc : list | tuple | None
        Names of channels or list of indices that should be designated
        MISC channels.
    include : list | None
        Event channels to include when creating annotations or a synthetic
        trigger channel.
    exclude : list | None
        Event channels to exclude when creating annotations or a synthetic
        trigger channel.
    %(preload)s
    channel_naming : str
        Channel naming convention for EEG channels.
    events_as_annotations : bool
        If True, create annotations from experiment events. If False,
        synthesize a trigger channel ``STI 014``.
    %(verbose)s

    Returns
    -------
    raw : instance of RawMffPy
        The raw object.

    Notes
    -----
    Only continuous MFF files are supported by this backend.
    """
    return RawMffPy(
        input_fname,
        eog,
        misc,
        include,
        exclude,
        preload,
        channel_naming,
        events_as_annotations=events_as_annotations,
        verbose=verbose,
    )


class RawMffPy(BaseRaw):
    """Raw object from EGI MFF file, using mffpy throughout.

    Both signal data and events are read via mffpy.  The mffpy ``Reader``
    handles calibration (GCAL auto-loaded) and unit conversion to Volts,
    so all MNE channel ``cal`` values are ``1.0`` for EEG.  PNS channel
    calibration is applied by mffpy's unit scaling, then by MNE's per-channel
    ``cal`` values set in :func:`_add_pns_channel_info`.
    """

    _extra_attributes = ("event_id",)

    @verbose
    def __init__(
        self,
        input_fname,
        eog=None,
        misc=None,
        include=None,
        exclude=None,
        preload=False,
        channel_naming="E%d",
        *,
        events_as_annotations=True,
        verbose=None,
    ):
        """Initialize a raw object from a continuous EGI MFF file.

        Parameters
        ----------
        input_fname : path-like
            Path to the MFF directory.
        eog : list | tuple | None
            Names of channels or list of indices that should be designated
            EOG channels.
        misc : list | tuple | None
            Names of channels or list of indices that should be designated
            MISC channels.
        include : list | None
            Event channels to include when creating annotations or a synthetic
            trigger channel.
        exclude : list | None
            Event channels to exclude when creating annotations or a synthetic
            trigger channel.
        %(preload)s
        channel_naming : str
            Channel naming convention for EEG channels.
        events_as_annotations : bool
            If True, create annotations from experiment events. If False,
            synthesize a trigger channel ``STI 014``.
        %(verbose)s
        """
        from mffpy import Reader

        input_fname = str(
            _check_fname(
                input_fname,
                "read",
                True,
                "input_fname",
                need_dir=True,
            )
        )
        logger.info(f"Reading EGI MFF Header from {input_fname}...")
        egi_info = _read_header_mffpy(input_fname)
        if eog is None:
            eog = []
        if misc is None:
            misc = np.where(np.array(egi_info["chan_type"]) != "eeg")[0].tolist()

        # mffpy loads GCAL automatically on Reader init.
        reader = Reader(input_fname)
        reader.set_unit("EEG", "V")
        if "PNSData" in reader.sampling_rates:
            reader.set_unit("PNSData", "V")

        logger.info("    Reading events via mffpy ...")
        egi_events, egi_info, mff_events = _read_mff_events_mffpy(
            input_fname, egi_info
        )

        # mffpy returns Volts directly, so MNE channel cals are 1.0 for EEG.
        cals = np.ones(len(egi_info["chan_type"]))

        logger.info("    Assembling measurement info ...")
        event_codes = egi_info["event_codes"]
        include = _triage_include_exclude(include, exclude, egi_events, egi_info)
        if egi_info["n_events"] > 0 and not events_as_annotations:
            logger.info('    Synthesizing trigger channel "STI 014" ...')
            if all(ch.startswith("D") for ch in include):
                # Support DIN1…DIN9, DI10…DI99, D100…D255 from parallel port.
                events_ids = []
                for ch in include:
                    while not ch[0].isnumeric():
                        ch = ch[1:]
                    events_ids.append(int(ch))
            else:
                events_ids = np.arange(len(include)) + 1
            egi_info["new_trigger"] = _combine_triggers(
                egi_events[[c in include for c in event_codes]], remapping=events_ids
            )
            self.event_id = dict(
                zip([e for e in event_codes if e in include], events_ids)
            )
            if egi_info["new_trigger"] is not None:
                egi_events = np.vstack([egi_events, egi_info["new_trigger"]])
        else:
            self.event_id = None
            egi_info["new_trigger"] = None
        assert egi_events.shape[1] == egi_info["last_samps"][-1]

        meas_dt_utc = egi_info["meas_dt_local"].astimezone(datetime.timezone.utc)
        info = _empty_info(egi_info["sfreq"])
        info["meas_date"] = _ensure_meas_date_none_or_dt(meas_dt_utc)
        info["utc_offset"] = egi_info["utc_offset"]
        info["device_info"] = dict(type=egi_info["device"])

        # Reuse the native MFF location/montage handling.
        ch_names, mon = _read_locs(input_fname, egi_info, channel_naming)
        ch_names.extend(list(egi_info["event_codes"]))
        n_extra = len(event_codes) + len(misc) + len(eog) + len(egi_info["pns_names"])
        if egi_info["new_trigger"] is not None:
            ch_names.append("STI 014")
            n_extra += 1

        ch_names.extend(egi_info["pns_names"])
        cals = np.concatenate([cals, np.ones(n_extra)])
        assert len(cals) == len(ch_names), (len(cals), len(ch_names))

        ch_coil = FIFF.FIFFV_COIL_EEG
        ch_kind = FIFF.FIFFV_EEG_CH
        chs = _create_chs(ch_names, cals, ch_coil, ch_kind, eog, (), (), misc)

        sti_ch_idx = [
            i
            for i, name in enumerate(ch_names)
            if name.startswith("STI") or name in event_codes
        ]
        for idx in sti_ch_idx:
            chs[idx].update(
                {
                    "unit_mul": FIFF.FIFF_UNITM_NONE,
                    "cal": cals[idx],
                    "kind": FIFF.FIFFV_STIM_CH,
                    "coil_type": FIFF.FIFFV_COIL_NONE,
                    "unit": FIFF.FIFF_UNIT_NONE,
                }
            )
        # PNS channels: mffpy returns Volts (set_unit above), so cal stays 1.0.
        # _add_pns_channel_info sets kind/coil_type/unit but we override cal=1.0.
        chs = _add_pns_channel_info(chs, egi_info, ch_names)
        for ch_name in egi_info["pns_names"]:
            chs[ch_names.index(ch_name)]["cal"] = 1.0

        info["chs"] = chs
        info._unlocked = False
        info._update_redundant()

        if mon is not None:
            info.set_montage(mon, on_missing="ignore")
            ref_idx = np.flatnonzero(np.isin(mon.ch_names, REFERENCE_NAMES))
            if len(ref_idx):
                ref_idx = ref_idx.item()
                ref_coords = info["chs"][int(ref_idx)]["loc"][:3]
                for chan in info["chs"]:
                    if chan["kind"] == FIFF.FIFFV_EEG_CH:
                        chan["loc"][3:6] = ref_coords

        egi_info["egi_events"] = egi_events
        egi_info["_mffpy_reader"] = reader

        keys = ("eeg", "sti", "pns")
        idx = dict()
        idx["eeg"] = np.where([ch["kind"] == FIFF.FIFFV_EEG_CH for ch in chs])[0]
        idx["sti"] = np.where([ch["kind"] == FIFF.FIFFV_STIM_CH for ch in chs])[0]
        idx["pns"] = np.where(
            [
                ch["kind"]
                in (FIFF.FIFFV_ECG_CH, FIFF.FIFFV_EMG_CH, FIFF.FIFFV_BIO_CH)
                for ch in chs
            ]
        )[0]
        if not np.array_equal(
            np.concatenate([idx[key] for key in keys]), np.arange(len(chs))
        ):
            raise ValueError(
                "Currently interlacing EEG and PNS channels is not supported"
            )
        egi_info["kind_bounds"] = [0]
        for key in keys:
            egi_info["kind_bounds"].append(len(idx[key]))
        egi_info["kind_bounds"] = np.cumsum(egi_info["kind_bounds"])
        assert egi_info["kind_bounds"][0] == 0
        assert egi_info["kind_bounds"][-1] == info["nchan"]
        first_samps = [0]
        last_samps = [egi_info["last_samps"][-1] - 1]

        annot = dict(onset=list(), duration=list(), description=list())

        if len(idx["pns"]):
            pns_samples = np.sum(egi_info["pns_sample_blocks"])
            eeg_samples = np.sum(egi_info["eeg_sample_blocks"])
            if pns_samples == eeg_samples - 1:
                warn("This file has the EGI PSG sample bug")
                annot["onset"].append(last_samps[-1] / egi_info["sfreq"])
                annot["duration"].append(1 / egi_info["sfreq"])
                annot["description"].append("BAD_EGI_PSG")
            elif pns_samples != eeg_samples:
                raise RuntimeError(
                    f"PNS samples ({pns_samples}) did not match EEG samples "
                    f"({eeg_samples})."
                )

        super().__init__(
            info,
            preload=preload,
            orig_format="single",
            filenames=[input_fname],
            first_samps=first_samps,
            last_samps=last_samps,
            raw_extras=[egi_info],
            verbose=verbose,
        )

        for first, prev_last in zip(
            egi_info["first_samps"][1:], egi_info["last_samps"][:-1]
        ):
            gap = first - prev_last
            assert gap >= 0
            if gap:
                annot["onset"].append((prev_last - 0.5) / egi_info["sfreq"])
                annot["duration"].append(gap / egi_info["sfreq"])
                annot["description"].append("BAD_ACQ_SKIP")

        if events_as_annotations:
            for code, samples in mff_events.items():
                if code not in include:
                    continue
                annot["onset"].extend(np.array(samples) / egi_info["sfreq"])
                annot["duration"].extend([0.0] * len(samples))
                annot["description"].extend([code] * len(samples))

        if len(annot["onset"]):
            self.set_annotations(Annotations(**annot))

    def close(self):
        """Close any open mffpy reader resources."""
        for extra in getattr(self, "_raw_extras", []):
            reader = extra.get("_mffpy_reader")
            if reader is None:
                continue
            for blob in getattr(reader, "_blobs", {}).values():
                blob.close()
            extra["_mffpy_reader"] = None
        super().close()

    def _read_segment_file(self, data, idx, fi, start, stop, cals, mult):
        """Read a chunk of data via mffpy.get_physical_samples."""
        logger.debug(f"Reading MFF {start:6d} ... {stop:6d} ...")

        egi_info = self._raw_extras[fi]
        reader = egi_info["_mffpy_reader"]
        sfreq = egi_info["sfreq"]
        bounds = egi_info["kind_bounds"]

        one = np.zeros((bounds[-1], stop - start))

        # Stim channels are precomputed; fill them directly.
        if isinstance(idx, slice):
            idx = np.arange(idx.start, idx.stop)
        stim_out = np.where((idx >= bounds[1]) & (idx < bounds[2]))[0]
        if len(stim_out):
            stim_in = idx[stim_out] - bounds[1]
            one[idx[stim_out], :] = egi_info["egi_events"][stim_in, start:stop]

        # Map from the MNE virtual timeline (which has gap samples set to -1)
        # to contiguous physical disk positions, then to mffpy time coordinates.
        disk_samps = egi_info["disk_samps"][start:stop]
        disk_use_idx = np.where(disk_samps > -1)[0]
        if not len(disk_use_idx):
            _mult_cal_one(data, one, idx, cals, mult)
            return

        disk_start = int(disk_samps[disk_use_idx[0]])
        disk_stop = int(disk_samps[disk_use_idx[-1]]) + 1
        t0 = disk_start / sfreq
        dt = (disk_stop - disk_start) / sfreq

        phys = reader.get_physical_samples(t0, dt)

        # EEG channels
        eeg_out = np.where(idx < bounds[1])[0]
        if len(eeg_out):
            eeg_in = idx[eeg_out]
            eeg_data, _ = phys["EEG"]
            one[eeg_in[:, np.newaxis], disk_use_idx] = eeg_data[eeg_in]

        # PNS channels
        pns_out = np.where((idx >= bounds[2]) & (idx < bounds[3]))[0]
        if len(pns_out) and "PNSData" in phys:
            pns_in = idx[pns_out] - bounds[2]
            pns_data, _ = phys["PNSData"]
            one[idx[pns_out, np.newaxis], disk_use_idx] = pns_data[pns_in]

        _mult_cal_one(data, one, idx, cals, mult)
