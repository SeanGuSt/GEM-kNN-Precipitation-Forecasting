function [expect, k_nn, observe] = GEM_kNN_Master(filename, targetDate, abPair, k, val2Forecast, userInput)
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% filename = string that leads to the file containing need data. Must be in
% the format year month day parameters, see excel files for examples.
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% targetDate = 3x1 vector containing the day to be forcasted.
% Must be in the format year month day.
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% abPair = 2x1 vector containing the ordered pair (a, b), read as a spans
% of b days. Used to build feature vectors. All feature vectors are
% comprised of a*b days for each parameter, with the first b days averaged
% together, followed by the next, and the next.
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% k = 1x1 integer. Used to determine how many of the nearest neighbors are
% used to produce the forecast.
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% val2Forecast = vector of the parameters you wish to forecast. They must 
% be numbered according to the order they appear in the table read from
% filename
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% userInput.daysForward = 1x1 integer. Used to determine how far into the
% fututre will be forecasted. If left empty, or not assigned, defaults to
% 30
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% userInput.lag_lead = 1x1 integer. Tells how many days before and after
% our target date on the same calendar day, but earlier years, we would
% like to consider a potential analog. If left empty, or not assigned,
% defaults to 7.
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% userInput.featVec_weights = num_var x 1 vector, the weight of each
% paramter towards the Euclidean distance. If left empty, or not assigned,
% defaults to a vector with each entry equal to 1/num_var
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% userInput.yearsBack2Trainab = 1x1 integer. Tells how many years back we
% want to go while training the (a, b) pair. in each year, the days
% indicated by userInput.days2Trainab in the relevant month will be run
% through GEM_kNN via train_abPair. If left empty, or not assigned,
% defaults to 25.
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
% userInput.days2Trainab = vector of integers. Tells which days of the
% month we wish to use in order to train the (a, b) pair during the years
% back indicated by userInput.yearsBack2Trainab. If left empty, or not
% assigned, defaults to 9:3:21 ([9, 12, 15, 18, 21]).
%\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
    options.daysForward = 30;
    if isfield(userInput, "daysForward") && ~isempty(userInput.daysForward)
        options.daysForward = userInput.daysForward;
    end
    options.lag_lead = 7;
    if isfield(userInput, "lag_lead") && ~isempty(userInput.lag_lead)
        options.lag_lead = userInput.lag_lead;
    end
    options.featVec_weights = 1;
    if isfield(userInput, "featVec_weights") && ~isempty(userInput.featVec_weights)
        options.featVec_weights = userInput.featVec_weights;
    end
    errCheck_targetDate(targetDate);%Check for something wrong with the targetDate
    data = table2array(readtable(filename));
    data = data(find(~isnan(data(:,1)),1):end, :);
    num_var = size(data, 2) - 3; %Number of variables (parameters) for feature vectors. Not indicative of which variable(s) will be forecast
    data_normalized = zeros(size(data) - [0, 3]);
    for i = 4:(num_var + 3)
        mean_data = mean(data(:, i));
        std_data = std(data(:, i));
        data_normalized(:, i-3) = (data(:, i) - mean_data)/std_data;
    end
    if numel(options.featVec_weights) == 1
        options.featVec_weights = ones(num_var, 1);
    end
    if find(val2Forecast <= 3, 1)
        val2Forecast = val2Forecast + 3;%The 1st parameter is in column 4 of data, the 2nd in column 5, and so on
    end
%   It is recommended that if you wish to train up an abPair for use, you
%   should instead use train_abPair_turbo. It will take longer, but you
%   will only need to run it once, after which you can run this program as
%   many times as you wish with the abPairs provided.
    if isempty(abPair)
        options.yearsBack2Trainab = 25;
        if isfield(userInput, "yearsBack2Trainab") && ~isempty(userInput.yearsBack2Trainab)
            options.yearsBack2Trainab = userInput.yearsBack2Trainab;
        end
        options.days2Trainab = 9:3:21;
        if isfield(userInput, "days2Trainab") && ~isempty(userInput.days2Trainab)
            options.days2Trainab = userInput.days2Trainab;
        end
        options.days2Trainab = errCheck_trainingValues(options, targetDate(2), targetDate(1), min(data(:, 1)));
        abPair = train_abPair(data, data_normalized, targetDate, k, val2Forecast, options);
    else
        errCheck_abPair(abPair);
    end
    [expect, k_nn, observe] = GEM_kNN(data, data_normalized, targetDate, abPair, k, val2Forecast, options);
end
function errCheck_abPair(abPair)
    a = abPair(1);
    b = abPair(2);
    switch true
        case a < 1
            error("We cannot have a feature vector comprised of %d spans. Please ensure a (the first entry of abPair) is a positive integer", a)
        case b < 1
            error("We cannot take the average of %d days. Please ensure b (the second entry of abPair) is a positive integer", b)
    end
end
function errCheck_targetDate(targetDate)
    y = targetDate(1);
    m = targetDate(2);
    d = targetDate(3);
    switch true
        case numel(targetDate) ~= 3
            error("Please ensure targetDate is comprised of exactly 3 nonnegative integers, ordered [year month day]")
        case m > 12
            error("As of this program's writing, there is no %s month.", iptnum2ordinal(m))
        case d > 31
            error("A month cannot have %d days. Please ensure the day value does not exceed 31.", d)
        case d == 31 && ismember(m, [2, 4, 6, 9, 11])
            error("The month given cannot have a 31st day.")
        case d == 30 && m == 2
            error("February cannot have a 30th, even on leap years.")
        case d == 29 && m == 2 && ~isLeapYear(y)
            error("February can only have a 29th on leaps years, which %d is not.", y)
    end
end
function days = errCheck_trainingValues(options, m, y, earliestYear)
    yearsBack = options.yearsBack2Trainab;
    days = options.days2Trainab;
    switch true
        case find(days > 31, 1)
            error("As of this program's writing, a month cannot have %d days", max(days))
        case any(find(days > 30, 1)) && ismember(m, [2, 4, 6, 9, 11])
            error("The month given cannot have a 31st day.")
        case ismember(29, days) && m == 2
            warning("Cannot find February 29ths in every year. For the purposes of training, February 28th will be looked at instead.")
            days(days == 29) = 28;
        case y - yearsBack <= earliestYear
            error("You do not have data available for %d or earlier. Please lower userInput.yearsBack2Trainab by at least %d.", earliestYear - 1, earliestYear - (y - yearsBack) + 1);
    end
end
function bool = isLeapYear(y)
    bool = mod(y, 4) == 0 && (mod(y,100) ~= 0 || mod(y, 400) == 0);
end