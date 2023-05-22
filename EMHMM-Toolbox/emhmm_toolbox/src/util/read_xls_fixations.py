import pandas as pd
import numpy as np

def read_xls_fixations(xlsname = None,opt = None): 
    # read_xls_fixations - read an EXCEL file with fixation data
    
    #  [data, subject_names, trial_names] = read_xls_fixations(xlsname, opt)
    
    # Expected header cells in the spreadsheet:
    #   SubjectID = subject ID
    #   TrialID   = trial ID for subject
    #   FixX      = fixation X-location
    #   FixY      = fixation Y-location
    #   FixD      = fixation duration in milliseconds (optional)
    
    # SubjectID and TrialID can be either strings or numbers.
    # FixX and FixY must be numbers.
    # FixD is a number (milliseconds).
    
    # Data will be separated by SubjectID and TrialID.
    # For each trial, fixations are assumed to be in sequential order.
    
    # INPUT
    #   xlsname - filename for the Excel spreedsheet (xls)
    #   opt     - options (not used)
    
    # OUTPUT
    #   data - data cell array
    #     data{i}         = i-th subject
    #     data{i}{j}      = ... j-th trial
    #     data{i}{j}(t,:) = ... [x y] location of t-th fixation
    #                        or [x y d] of t-th fixation (location & duration)
    
    #   The subject/trials will be assigned numbers according to their order in the spreadsheet.
    #   the following two outputs contain the original subject names and trial IDs:
    #     subject_names{i} - the subject ID in the spreadsheet for the i-th subject
    #     trial_names{i}{j} - the j-th trial ID for i-th subject
        
    # ---
    # Eye-Movement analysis with HMMs (emhmm-toolbox)
    # Copyright (c) 2017-01-13
    # Antoni B. Chan, Janet H. Hsiao, Tim Chuk
    # City University of Hong Kong, University of Hong Kong
    
        
    print('Reading %s\n' % (xlsname))
    # read the XLS file
    
    rdata = pd.read_excel(xlsname)
    
    # get the headers
    headers = rdata.columns.values
    
    # find the header indices
    SID = rdata.columns.get_loc('SubjectID') if 'SubjectID' in headers else -1
    TID = rdata.columns.get_loc('TrialID') if 'TrialID' in headers else -1
    FX  = rdata.columns.get_loc('FixX') if 'FixX' in headers else -1
    FY = rdata.columns.get_loc('FixY') if 'FixY' in headers else -1
    FD =  rdata.columns.get_loc('FixD') if 'FixD' in headers else -1
    
    if len(SID) != 1:
        raise Exception('error with SubjectID')
    
    print('- found SubjectID in column %d\n' % (SID))
    if len(TID) != 1:
        raise Exception('error with TrialID')
    
    print('- found TrialID in column %d\n' % (TID))
    if len(FX) != 1:
        raise Exception('error with FixX')
    
    print('- found FixX in column %d\n' % (FX))
    if len(FY) != 1:
        raise Exception('error with FixY')
    
    print('- found FixY in column %d\n' % (FY))
    if len(FD) == 1:
        print('- found FixD in column %d\n' % (FD))
    else:
        if len(FD) > 1:
            raise Exception('error with FixD -- to many columns')
    
    
    # initialize names and trial names
    sid_names = np.array([])
    sid_trials = np.array([])
    data = np.array([])

    # read data
    for i in np.arange(2, rdata.shape[1] - 1).reshape(-1):
        mysid = rdata[i, SID]
        mytid = rdata[i, TID]
        
        if isinstance(rdata[i, FX], str):
            raise Exception('Value for FixX is text, not a number.')
        if isinstance(rdata[i, FY], str):
            raise Exception('Value for FixY is text, not a number.')
        
        myfxy = np.array([rdata[i, FX], rdata[i, FY]])
        
        if len(FD) == 1:
            # include duration if available
            if isinstance(rdata[i, FD], str):
                raise Exception('Value for FixD is text, not a number.')
            myfxy = np.array([myfxy, rdata[i, FD]])
        
        if True:
            mysid = str(mysid)
        
        if True:
            mytid = str(mytid)
        
        # find subject
        s = np.where(str(mysid) == str(sid_names))[0]
        
        if len(s) == 0:
            # new subject
            sid_names = np.append(sid_names, mysid)
            sid_trials = np.append(sid_trials, np.array([]))
            s = len(sid_names) - 1
            data = np.append(data, np.array([]))
            data[s, 0] = np.array([])
        
        # find trial
        t = np.where(str(mytid) == str(sid_trials[s]))[0]
        
        if len(t) == 0:
            sid_trials[s] = np.append(sid_trials[s], mytid)
            t = len(sid_trials[s]) - 1
            data[s, 0] = np.append(data[s, 0], [])
        
        # put fixation
        data[s, 0][t, 0] = np.append(data[s, 0][t, 0], myfxy)
    
    print('- found %d subjects:\n' % (len(sid_names)))
    print('%s ' % (sid_names[:]))
    print('\n' % ())
    for i in np.arange(1,len(data)+1).reshape(-1):
        print('    * subject %d had %d trials\n' % (i,len(data[i])))
    
    return data,sid_names,sid_trials