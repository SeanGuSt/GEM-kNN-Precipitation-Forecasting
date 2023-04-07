function [expect, k_nn, observe] = GEM_kNN(data, data_normalized, targetDate, abPair, k, val2Forecast, options)
%   Used through GEM_kNN_Master, which checks the validity of the values
%   given and normalizes the data. Explanation of what all variables do is
%   given in the GEM_kNN_Master comments.
%
%   This program, meanwhile, builds both historical and target feature
%   vectors and takes the Euclidean Distance of the former from the latter.
%   Once this is done, the program takes note of the k analogs with the
%   lowest distances, then finds the mean of the parameters noted in
%   val2Forecast and takes the weighted sum of the mean occurence for each
%   over the proceeding options.daysForward days, storing it in expect,
%   while the individual k analogs will be placed in k_nn. If able to (only
%   during hindcast testing is this possible,) 
%   the
    a = abPair(1);
    b = abPair(2);
    ab = prod(abPair);
    y = targetDate(1);
    m = targetDate(2);
    d = targetDate(3);
    daysForward = options.daysForward;
    lag_lead = options.lag_lead;
    featVec_weights = options.featVec_weights;
%   First, find all days which share the calendar day and month of the
%   target date, including the target date itself
    dateIndex_targetCalendarDay = find(data(:, 1)<=y &...
                                     data(:, 2)==m &...
                                     data(:, 3)==d);
%   Assign the actual target date index
    dateIndex_target = dateIndex_targetCalendarDay(end);
    dateIndex_earliestsameDay = dateIndex_targetCalendarDay(1);
%   Then remove it so it isn't considered a potential analogue
    dateIndex_targetCalendarDay(dateIndex_targetCalendarDay + lag_lead + daysForward >= dateIndex_target) = [];
%   If this year doesn't have enough days to build such an early feature
%   vector, omit it
    dateIndex_targetCalendarDay(dateIndex_targetCalendarDay - ab - lag_lead < 1) = [];
    if isempty(dateIndex_targetCalendarDay)
        error("Reduce the size of either lag_lead or abPair to ensure potential analogue dates can be found. To be specific, your current parameters are attempting to index a date %d days earlier than you have available", lag_lead + ab - dateIndex_earliestsameDay);
    end
%   Decide the window of days around each analog day that will also be
%   considered, then collect all those days
    lag = -lag_lead;
    lead = lag_lead;
    windowSize = 2*lag_lead+1;
    windowSet = 1:windowSize;
    num_sddy = numel(dateIndex_targetCalendarDay);
    dateIndex_analog = zeros(1, windowSize*num_sddy);
    for i = 1:num_sddy
        dateIndex_analog(windowSet) = dateIndex_targetCalendarDay(i) + (lag:lead);
        windowSet = windowSet + windowSize;
    end
%   Build feature vector of target date using the (a, b) pair provided
    num_analog = numel(dateIndex_analog);
    num_var = size(data, 2) - 3;
    zero2ab1 = 0:(ab - 1);
    featVec = data_normalized(dateIndex_target - zero2ab1, :);
    featVec_target = buildFeatureVector(featVec, a, b, num_var);
    ed = 1:num_analog;
%   Do the same for each analog date, then calculate its distance from the
%   target date's feature vector
    for i = 1:num_analog
        featVec = data_normalized(dateIndex_analog(i) - zero2ab1, :);
        featVec_analog = buildFeatureVector(featVec, a, b, num_var);
        ed(i) = sum( (featVec_target - featVec_analog).^2, 1)*featVec_weights;
    end
%   Find the k nearest neighbors and take their weighted sum
    [~, ed_sorted] = sort(ed);
    if isempty(k), k = floor(sqrt(num_analog)); end
    if k > num_analog
        error("k exceeds the number of analogs available. Please lower k or include more data")
    end
    k_daysChosen = dateIndex_analog(ed_sorted(1:k));
    k_weights = 1./(1:k);
    k_weights = k_weights/sum(k_weights);
%     k_weights = k^2 - (0:(k-1)).^2;
%     k_weights = k_weights/(k*(k+1)*(4*k-1)/6);
%     k_weights = 1./sqrt(ed(ed_sorted(1:k)));
    k_nn = zeros(k, numel(val2Forecast));
    expect = zeros([1, numel(val2Forecast)]);
    df = 1:daysForward;
    for i = 1:k
        k_nn(i, :) = sum(data(k_daysChosen(i) + df, val2Forecast), 1);
        expect = expect + k_nn(i, :)*k_weights(i);
    end
    if nargout > 2
        try observe = sum(data(dateIndex_target + df, val2Forecast), 1);
        catch
            warning("Not enough data to generate the desired observed results. Assigning observe = -1");
            observe = -1;
        end
    end
    
end

function fv = buildFeatureVector(featVec, a, b, num_var)
    b_vec = 0:b:(a*b);
    fv = zeros(a, num_var);
    for i = 1:a
        fv(i, :) = sum(featVec((b_vec(i)+1):b_vec(i+1), :), 1);
    end
end