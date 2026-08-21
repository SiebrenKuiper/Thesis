clear; clc;

main_dir        = 'D:/Research/MS_Thesis/Study2/Output';
IMG_DIR         = fullfile(main_dir, 'Image');
CLEANED_DIR     = fullfile(IMG_DIR, 'cleaned');
LSS_DIR         = fullfile(IMG_DIR, 'LSS', 'LSS_run01');
Quest_DATA      = fullfile(main_dir, 'Questionnaire', 'ques_data.csv');
NPS_DIR         = fullfile(IMG_DIR, 'NPS');
OUTPUT_DIR      = fullfile(IMG_DIR, 'MMM', 'mediation_run01');
group_mask      = 'D:/Research/MS_Thesis/Study2/Output/Image/masks/group_mask_belief_run01.nii';

cd('D:/Research/MS_Thesis/Study2/Analysis')
if exist(OUTPUT_DIR, 'dir')
    if strcmp(pwd, OUTPUT_DIR)
        cd(main_dir);
    end
    rmdir(OUTPUT_DIR, 's');
end
mkdir(OUTPUT_DIR);

TRIM_NII_DIR = fullfile(OUTPUT_DIR, 'trimmed_nii');
if ~exist(TRIM_NII_DIR, 'dir'), mkdir(TRIM_NII_DIR); end

quest_data     = readtable(Quest_DATA);
subid_num_vec  = double(string(quest_data.SubID));
formatted_keys = arrayfun(@(x) sprintf('%02d', x), subid_num_vec, 'UniformOutput', false);
group_num      = double(string(quest_data.Group) == "pain");
group_map      = containers.Map(formatted_keys, group_num);

tsv_files = dir(fullfile(CLEANED_DIR, 'sub-*_task-belief_run-01_events.tsv'));
n_subs    = numel(tsv_files);

X = cell(n_subs, 1);  Y = cell(n_subs, 1);  M = cell(n_subs, 1);
Group_vec = zeros(n_subs, 1);  NPS_vec = zeros(n_subs, 1);
subcount = 0;  skip_count = 0;

for i = 1:n_subs
    tsv_name     = tsv_files(i).name;
    subid        = regexp(tsv_name, 'sub-\d+', 'match', 'once');
    subid_padded = sprintf('%02d', str2double(regexp(tsv_name, '(?<=sub-)\d+', 'match', 'once')));

    nii_file = fullfile(LSS_DIR, sprintf('%s_task-belief_run-01_condition_feedback_lss.nii', subid));
    nps_file = fullfile(NPS_DIR, sprintf('sub-%s_task-belief_run-01_NPS.tsv', subid_padded));

    t          = readtable(fullfile(tsv_files(i).folder, tsv_name), 'FileType', 'text', 'Delimiter', '\t');
    t_feedback = t(strcmp(t.trial_type_mediation, 'feedback'), :);

    if isempty(t_feedback) || ~isfile(nii_file) || ~isfile(nps_file) || ~group_map.isKey(subid_padded)
        skip_count = skip_count + 1;
        continue;
    end

    EE_raw = double(t_feedback.EE);
    UP_raw = double(t_feedback.Update);

    EE_sub = EE_raw(1:end-1);
    UP_sub = UP_raw(1:end-1);

    nii_info = niftiinfo(nii_file);
    nii_data = niftiread(nii_info);

    nii_data_trim = nii_data(:, :, :, 1:end-1);

    if numel(EE_sub) ~= size(nii_data_trim, 4) || numel(UP_sub) ~= size(nii_data_trim, 4)
        warning('Mismatch %s: EE=%d, UP=%d, IMG=%d. Skipped.', ...
            subid, numel(EE_sub), numel(UP_sub), size(nii_data_trim,4));
        skip_count = skip_count + 1;
        continue;
    end

    trimmed_nii_file = fullfile(TRIM_NII_DIR, ...
        sprintf('%s_task-belief_run-01_condition_feedback_lss_trimlast.nii', subid));

    nii_info_trim = nii_info;
    nii_info_trim.ImageSize = size(nii_data_trim);
    niftiwrite(nii_data_trim, trimmed_nii_file, nii_info_trim, 'Compressed', false);


    EE_sub = EE_sub - mean(EE_sub, 'omitnan');
    UP_sub = UP_sub - mean(UP_sub, 'omitnan');

    nps_table = readtable(nps_file, 'FileType', 'text', 'Delimiter', '\t');

    subcount = subcount + 1;
    X{subcount}         = EE_sub;
    Y{subcount}         = UP_sub;
    M{subcount}         = trimmed_nii_file;
    Group_vec(subcount) = group_map(subid_padded);
    NPS_vec(subcount)   = mean(nps_table{:,1}, 'omitnan');
end

X = X(1:subcount);  Y = Y(1:subcount);  M = M(1:subcount);
Group_vec = Group_vec(1:subcount) * 2 - 1;
NPS_vec   = NPS_vec(1:subcount) - mean(NPS_vec(1:subcount));

fprintf('Included subjects: %d, Skipped: %d\n', subcount, skip_count);

analyses = {
    'nomod',  [],         'No Moderator';
    'Group',  Group_vec,  'Moderator: Group';
    'NPS',    NPS_vec,    'Moderator: centered NPS';
};

SETUP.mask           = group_mask;
SETUP.preprocX       = 0;
SETUP.preprocY       = 0;
SETUP.preprocM       = 0;
SETUP.wh_is_mediator = 'M';

for a = 1:size(analyses, 1)
    mod_name  = analyses{a, 1};
    L2M_cur   = analyses{a, 2};
    mod_label = analyses{a, 3};

    result_subdir = fullfile(OUTPUT_DIR, mod_name);
    if ~exist(result_subdir, 'dir'), mkdir(result_subdir); end
    cd(result_subdir);

    if isempty(L2M_cur)
        mediation_brain_multilevel(X, Y, M, SETUP, ...
            'nopreproc', ...
            'names', {'EE', 'Update', 'BOLD'}, ...
            'boot', 'bootsamples', 10000, 'doplots', 'dosave');
    else
        mediation_brain_multilevel(X, Y, M, SETUP, ...
            'nopreproc', 'L2M', L2M_cur, ...
            'names', {'EE', 'Update', 'BOLD'}, ...
            'boot', 'bootsamples', 10000, 'doplots', 'dosave');
    end

    SETUP_cur = mediation_brain_corrected_threshold('fdr');

    mediation_brain_results('a', 'thresh', SETUP_cur.fdr_p_thresh, 'size', 5, 'slices', 'tables', 'names', 'save');
    mediation_brain_results('b',  'thresh', SETUP_cur.fdr_p_thresh, 'size', 5, 'slices', 'tables', 'names', 'save');
    mediation_brain_results('ab', 'thresh', SETUP_cur.fdr_p_thresh, 'size', 5, 'slices', 'tables', 'names', 'save');
    fprintf('Completed analysis: %s\n', mod_label);
end