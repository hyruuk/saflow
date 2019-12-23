import os
import os.path as op
from saflow_utils import find_rawfile, saflow_preproc
from saflow_params import FOLDERPATH, SUBJ_LIST, BLOCS_LIST, REPORTS_PATH


if __name__ == "__main__":
	# create report path
	try:
		os.mkdir(REPORTS_PATH)
	except:
		print('Report path already exists.')
	for subj in SUBJ_LIST:
		for bloc in BLOCS_LIST:
			filepath, filename = find_rawfile(subj, bloc, BIDS_PATH)
			save_pattern =  op.join(FOLDERPATH + filepath, filename[:-3] + '_preproc_raw.fif.gz')
			report_pattern = op.join(REPORTS_PATH, filename[:-3] + '_preproc_report.html')
			full_filepath = BIDS_PATH + filepath + filename
			saflow_preproc(full_filepath, save_pattern, report_pattern)
