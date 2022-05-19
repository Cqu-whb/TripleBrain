% -----------------------------------------------------------------------------------------------------------------------------------------------------%
% @编写：wtx
% @更新日期：2022/1/1
% @概述：读取每个bin文件的事件流列表的函数
% @备注：参考原函数略作修改
% -----------------------------------------------------------------------------------------------------------------------------------------------------%

function TD = Read_N_MNIST(filename)
eventData = fopen(filename);
evtStream = fread(eventData);
fclose(eventData);

TD.x    = evtStream(1:5:end)+1; %pixel x address, with first pixel having index 1
TD.y    = evtStream(2:5:end)+1; %pixel y address, with first pixel having index 1
TD.p    = bitshift(evtStream(3:5:end), -7)+1; %polarity, 1 means off, 2 means on
TD.ts   = bitshift(bitand(evtStream(3:5:end), 127), 16); %time in microseconds
TD.ts   = TD.ts + bitshift(evtStream(4:5:end), 8);
TD.ts   = TD.ts + evtStream(5:5:end);
return