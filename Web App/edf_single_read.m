
file = "static/input.edf";

[~,record]= edfread1(file);

fs = 256;
f_low = 14;
f_high = 70;
order = 4;
[b, a] = butter(order, [f_low, f_high]/(fs/2), 'bandpass');


data = vertcat(vertcat(vertcat(record(1:7,:),record(9:13,:),record(15:20,:),record(22:22,:))))';

data = filter(b, a, data);

save("static/data.mat", 'data');