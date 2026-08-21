dataFolder = 'D:/Research/MS_Thesis/Study2/Data/BEH/DOT';
matFiles = dir(fullfile(dataFolder, '*.mat'));
for k = 1:length(matFiles)
    try
        baseFileName = matFiles(k).name;
        fullFileName = fullfile(matFiles(k).folder, baseFileName);
        DATA = load(fullFileName);
        mainStruct = DATA.DATA.main;
        dataCell = {[]};
        trialCount = 0;

        for m = 1:numel(mainStruct)
            results = mainStruct(m).results;
            if isfield(results, 'response') && ~isempty(results.response)
                trial_num = length(results.response(:));
            else
                trial_num = 1;
            end
            for t = 1:trial_num
                trialCount = trialCount + 1;
                resp_val = NaN;
                rt_val = NaN;
                correct_val = NaN;
                t0time_val = NaN;
                confrating_val = NaN;
                confrt_val = NaN;
                conftime_val = NaN;
                diff_val = NaN;
                resptime_val = NaN;
                endtime_val = NaN;
                if isfield(results, 'response') && t <= length(results.response)
                    resp_val = results.response(t) - 1;
                end
                if isfield(results, 'rt') && t <= length(results.rt)
                    rt_val = results.rt(t);
                end
                if isfield(results, 'correct') && t <= length(results.correct)
                    correct_val = results.correct(t);
                end
                if isfield(results, 'T0Time') && t <= length(results.T0Time)
                    t0time_val = results.T0Time(t);
                end
                if isfield(results, 'responseConf') && t <= length(results.responseConf)
                    confrating_val = results.responseConf(t);
                end
                if isfield(results, 'rtConf') && t <= length(results.rtConf)
                    confrt_val = results.rtConf(t);
                end
                if isfield(results, 'contrast') && t <= length(results.contrast)
                    diff_val = results.contrast(t);
                end
                if isfield(results, 'shichang') && t <= length(results.shichang)
                    conftime_val = results.shichang(t)-2;
                    resptime_val = results.shichang(t)-5;
                    endtime_val = results.shichang(t) + results.T0Time(t);
                end
                dataCell{trialCount} = [resp_val, rt_val, correct_val, t0time_val, ...
                                       confrating_val, confrt_val, diff_val,...
                                       conftime_val, resptime_val,endtime_val];
            end
        end
        if trialCount > 0
            dataMatrix = cell2mat(dataCell');
            data_table = table( ...
                dataMatrix(:,1), ...  % Response
                dataMatrix(:,2), ...  % Response RT
                dataMatrix(:,3), ...  % Correct
                dataMatrix(:,4), ...  % Trial Start Time
                dataMatrix(:,5), ...  % Confidence Rating
                dataMatrix(:,6), ...  % Confidence RT
                dataMatrix(:,7), ...  % Difficulty
                dataMatrix(:,8), ...  % Confidence Time
                dataMatrix(:,9), ...  % Response Time
                dataMatrix(:,10), ... % Trial End Time
                'VariableNames', ...
                {'Resp', 'RespRT', 'Correct', 'TrialStart', ...
                 'ConfRating', 'ConfRT', 'Difficulty', 'ConfTime', ...
                 'RespTime','EndTime'});
            [filepath, name, ~] = fileparts(fullFileName);
            idStr = regexp(name, '\d+', 'match');
            if ~isempty(idStr)
                idNum = str2double(idStr{1});
                newIdStr = sprintf('%02d', idNum);
                newName = regexprep(name, '\d+', newIdStr);
            else
                newName = name;
            end
            outputFileName = fullfile(filepath, [newName '.csv']);
            writetable(data_table, outputFileName);
            fprintf('Processed file %d/%d: %s (%d valid trials) -> Saved as: %s\n', ...
                k, length(matFiles), baseFileName, trialCount, [newName '.csv']);
        else
            fprintf('Warning: %s contains no valid trial data. Skipping save.\n', baseFileName);
        end
    catch ME
        fprintf('Error: Failed to process file %d/%d - %s: %s\n', k, length(matFiles), baseFileName, ME.message);
        continue;
    end
end