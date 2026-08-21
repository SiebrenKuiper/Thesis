clear; clc; close all;

PREPROC_DIR = 'D:/Research/MS_Thesis/Study2/Output/Image/cleaned';
OUTPUT_DIR  = 'D:/Research/MS_Thesis/Study2/Output/Image/NPS';
if exist(OUTPUT_DIR, 'dir')
    rmdir(OUTPUT_DIR, 's');
end
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end


file_pattern = '*_task-belief_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii*';
all_files = dir(fullfile(PREPROC_DIR, '**', file_pattern));

for k = 1:length(all_files)
    img_path = fullfile(all_files(k).folder, all_files(k).name);
    [~, fname, ext] = fileparts(all_files(k).name);
    sub_token = regexp(fname, 'sub-(\d+)', 'tokens', 'once');
    run_token = regexp(fname, 'run-(\d+)', 'tokens', 'once');
    if isempty(sub_token) || isempty(run_token)
        warning('Filename does not match expected pattern: %s\n', all_files(k).name);
        continue;
    end
    
    sub_id = str2double(sub_token{1});
    run_id = str2double(run_token{1});
    sub_str = sprintf('sub-%02d', sub_id);
    run_str = sprintf('run-%02d', run_id);
    
    fprintf('Processing: %s, %s\n', sub_str, run_str);
    try
        nps_vals = apply_nps(img_path);
        nps_vals = nps_vals{1}
    catch ME
        warning('Error occurred while processing %s: %s\n', all_files(k).name, ME.message);
        continue;
    end
    out_filename = sprintf('%s_task-belief_%s_NPS.tsv', sub_str, run_str);
    out_file = fullfile(OUTPUT_DIR, out_filename);
    writematrix(nps_vals, out_file, 'Delimiter', 'tab', 'FileType', 'text');
    fprintf('NPS saved: %s\n', out_file);
end

fprintf('\nAll done!\n');