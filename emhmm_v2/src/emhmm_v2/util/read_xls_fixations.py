import pandas as pd

def read_xls_fixations(xlsname, opt=None):
    # Read the Excel file
    xls_data = pd.read_excel(xlsname)
    
    # Get the headers
    headers = list(xls_data.columns)
    
    # Find the header indices
    SID = headers.index('SubjectID')
    TID = headers.index('TrialID')
    FX = headers.index('FixX')
    FY = headers.index('FixY')
    FD = headers.index('FixD') if 'FixD' in headers else None
    
    if SID is None or TID is None or FX is None or FY is None:
        raise ValueError('Missing header(s) in the Excel file.')
    
    print(f'- found SubjectID in column {SID}')
    print(f'- found TrialID in column {TID}')
    print(f'- found FixX in column {FX}')
    print(f'- found FixY in column {FY}')
    
    if FD is not None:
        print(f'- found FixD in column {FD}')
    
    # Initialize variables
    sid_names = []
    sid_trials = []
    data = []
    
    # Read data
    for i, row in xls_data.iterrows():
        mysid = str(row[SID])
        mytid = str(row[TID])
        myfxy = [row[FX], row[FY]]
        
        if FD is not None:
            myfxy.append(row[FD])
        
        # Find subject
        s = sid_names.index(mysid) if mysid in sid_names else None
        if s is None:
            # New subject
            sid_names.append(mysid)
            sid_trials.append([])
            s = len(sid_names) - 1
            data.append([])
        
        # Find trial
        t = sid_trials[s].index(mytid) if mytid in sid_trials[s] else None
        if t is None:
            sid_trials[s].append(mytid)
            t = len(sid_trials[s]) - 1
            data[s].append([])
        
        # Put fixation
        data[s][t].append(myfxy)
    
    print(f'- found {len(sid_names)} subjects:')
    print(' '.join(sid_names))
    for i in range(len(data)):
        print(f'  * subject {i+1} had {len(data[i])} trials')
    
    return data, sid_names, sid_trials
