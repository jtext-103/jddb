from .file_repo import mds_dumper
from .file_repo import file_repo
import numpy as np

non_shot_list = np.load("./Non-disruptive_shot.npy")
dis_shot_list = np.load("./Disruptive_shot.npy")
shot_list = list(np.hstack((dis_shot_list, non_shot_list)))
file_re = file_repo.FileRepo("C:\AI\JDDB\jddb\Examples\FileRepo\$shot_2$\$shot_1$")
tag_list = [
    r"\ip", r"\bt", r"\vl", r"\Iohp", r"\Ivfp", r"\Ihfp",
    r"\polaris_den_v01", r"\polaris_den_v02", r"\polaris_den_v03", r"\polaris_den_v04", r"\polaris_den_v05",
    r"\polaris_den_v06", r"\polaris_den_v07", r"\polaris_den_v08", r"\polaris_den_v09", r"\polaris_den_v10",
    r"\polaris_den_v11", r"\polaris_den_v12", r"\polaris_den_v13", r"\polaris_den_v14", r"\polaris_den_v15",
    r"\polaris_den_v16", r"\polaris_den_v17",
    r"\dx", r"\dy",
    r"\vs_c3_aa001", r"\vs_c3_aa002", r"\vs_c3_aa003", r"\vs_c3_aa004", r"\vs_c3_aa005", r"\vs_c3_aa006",
    r"\vs_c3_aa007", r"\vs_c3_aa008", r"\vs_c3_aa009", r"\vs_c3_aa010", r"\vs_c3_aa011", r"\vs_c3_aa012",
    r"\vs_c3_aa013", r"\vs_c3_aa014", r"\vs_c3_aa015", r"\vs_c3_aa016", r"\vs_c3_aa017", r"\vs_c3_aa018",
    r"\sxr_cc_033", r"\sxr_cc_034", r"\sxr_cc_035", r"\sxr_cc_036", r"\sxr_cc_037", r"\sxr_cc_038",
    r"\sxr_cc_039", r"\sxr_cc_040", r"\sxr_cc_041", r"\sxr_cc_042", r"\sxr_cc_043", r"\sxr_cc_044",
    r"\sxr_cc_045", r"\sxr_cc_046", r"\sxr_cc_047", r"\sxr_cc_048", r"\sxr_cc_049", r"\sxr_cc_050",
    r"\sxr_cc_051", r"\sxr_cc_052", r"\sxr_cc_053", r"\sxr_cc_054", r"\sxr_cc_055", r"\sxr_cc_056",
    r"\sxr_cc_057", r"\sxr_cc_058", r"\sxr_cc_059", r"\sxr_cc_060", r"\sxr_cc_061", r"\sxr_cc_062",
    r"\MA_TOR1_R01", r"\MA_TOR1_R02", r"\MA_POL2_P01", r"\MA_POL2_P02",
    r"\exsad1", r"\exsad4", r"\exsad7", r"\exsad10"
]
dump = mds_dumper.MDSDumper('222.20.94.136', 'jtext')
dump.dumper(file_re, shot_list, tag_list)
