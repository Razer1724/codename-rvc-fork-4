# RVC stock discs
from .mpd_msd_combined import MPD_MSD_Combined

# [mpd, msd] core based
from .mpd_msd_mrd_combined import MPD_MSD_MRD_Combined
from .mpd_msd_mrd_univhd_combined import MPD_MSD_MRD_UnivHD_Combined

# [mpd, sbd] core based
from .mpd_sbd_mrd_combined import MPD_MSD_MRD_Combined as MPD_SBD_MRD_Combined

# Avocodo core [CoMBD, SBD] based
from .combd_sbd_combined import CoMBD_SBD_Combined
from .combd_sbd_mrd_combined import CoMBD_SBD_MRD_Combined
from .combd_sbd_univhd_combined import CoMBD_SBD_UnivHD_Combined

# Multi-Domain
from .hmdd import HolisticMultiDomainDiscriminator