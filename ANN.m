clc
clear all
tic
load USA.mat
NSE=zeros(1,1);
Files = dir('*.xlsx');
i=1;
%% Sorting recession events from the streamflow time-series
for i=1:273
    i
    store_recession = cell(5,5);
    Data= USA_Q{i,1:4};
    flow=Data(:,4);
    day=Data(:,3);
    month= Data(:,2);
    year= Data(:,1);
    [m n]=size(flow);
    rec_eve = [];
    rise_imp= [];
    rec_imp= [];
    date_imp=[];
    date_prev=[];
    flow_prev=[];
    month_imp=[];
    year_imp=[];
    [m n]=size(Data);
    j=1;
    for a=123:m-1 
        if flow(a+1)<flow(a)
            rec_imp1=flow(a);
            date_imp1=day(a);         
            month_imp1= month(a);
            year_imp1= year(a);
            rec_imp= vertcat(rec_imp,rec_imp1);
            date_imp= vertcat(date_imp,date_imp1);   
        else
            rec_imp1=flow(a);
            rec_imp= vertcat(rec_imp,rec_imp1);
            date_imp1=day(a);   
            date_imp= vertcat(date_imp,date_imp1);   
            if length(rec_imp)>10
                in_s=(a-length(rec_imp)-119);%%previous input indices coreesponding to past 120 days of discharge for a recession event
                ed= (a-length(rec_imp)+1); % index corresponding to peak dischage value
                date_prev=day(in_s:(ed)); %past date
                month_imp=month(in_s:(ed));
                year_imp=year(in_s:(ed));
                flow_prev= flow(in_s:ed); 
                Q_zero{j,1}=rec_imp(1);
                Q_zero{j,2}=date_imp(1);
                store_recession{j,1}= rec_imp(2:end);
                store_recession{j,2}= date_imp(2:end);
                store_recession{j,3}= month_imp;
                store_recession{j,4}= year_imp;
                store_recession{j,5}=date_prev;
                store_recession{j,6}=(flow_prev); 
                j = j+1;
                rec_imp =[];
                date_imp= [];
                date_prev=[];
                flow_prev=[];
                month_imp= [];
                year_imp= [];
                in_s=[];
                ed=[];
            end
            rec_imp =[];
            date_imp= [];
            date_prev=[];
            flow_prev=[];
            month_imp= [];
            year_imp= [];
            in_s=[];
            ed=[];
        end
        
    end
    %% For last event if streamflow is decreasing till last data point 
    if length(rec_imp)>10
        in_s=(a-length(rec_imp)-119);%previous input indices corresponding to past 120 days including peak
        ed= (a-length(rec_imp)+1);
        date_prev=day(in_s:(ed));
        month_imp=month(in_s:(ed));
        year_imp=year(in_s:(ed));
        flow_prev=flow(in_s:(ed));
        Q_zero{j,1}=rec_imp(1);
        Q_zero{j,2}=date_imp(1);
        store_recession{j,1}= rec_imp(2:end);
        store_recession{j,2}= date_imp(2:end);
        store_recession{j,3}= month_imp;
        store_recession{j,4}= year_imp;
        store_recession{j,5}=date_prev;
        store_recession{j,6}=flow_prev;
        rec_imp =[];
        date_imp= [];
        date_prev=[];
        flow_prev=[];
        month_imp= [];
        year_imp= [];
    end
    %% Training and testing
    total_data= size(store_recession,1);
    for p=1:total_data
        data_input(:,p) =store_recession{p,6}(:);  % Previous 120 days of daily discharge values
        data_output(1:10,p) =store_recession{p,1}(1:10) ; % Recession event lasting for 10 days
    end  
    tr_len=floor(0.6*(size(data_input,2)));% 60 of data for training and validation
    % Normalization of input data
    train_input=data_input(:,1:tr_len);
    input_mean=mean(train_input(:));
    input_sd= std(train_input(:));
    norm_input=(data_input-input_mean)/input_sd; 
    % Normalization of output data
    train_output=data_output(:,1:tr_len);
    output_mean=mean(train_output(:));
    output_sd= std(train_output(:));
    norm_output=(data_output-output_mean)/output_sd; 
    H=87; %No. of Neurons considered in the hidden layer =2/3*(sum of number of input and output variabless)
    net=feedforwardnet(H);
    data_length = size(data_output);
    leng= data_length(2);
    net.divideFcn='divideblock';% Dividing the data in to training, validation and testing
    net.divideParam.trainRatio = 40/100;
    net.divideParam.valRatio   = 20/100;
    net.divideParam.testRatio  = 40/100;
    net.layers{1}.transferFcn='logsig';
    net.trainFcn= 'trainlm';
    net = configure(net,norm_input,norm_output);
    rng(0)
    IW = 0.01*randn(H,121); %Initialization of weights
    b1 = 0.01*randn(H,1);
    LW = 0.01*randn(10,H);
    b2 = 0.01*randn(10,1);
    net.IW{1,1} = IW;
    net.b{1,1} = b1;
    net.LW{2,1} = LW;
    net.b{2,1} = b2;
    [net,tr(i)]=train(net,norm_input,norm_output); % Training the network
    norm_testX = norm_input(:,tr(i).testInd);
    norm_testY = net(norm_testX); %Output prediction using trained network
    testY= norm_testY*output_sd+output_mean;
    obsY = data_output(:,tr(i).testInd);
    %% Calculation of NSE for different recession days (Day 1 to Day 10)
    for nse_day=1:size(testY,2)
        store_recession{1,12}(nse_day,1)=obsY(1,nse_day);
        store_recession{2,12}(nse_day,1)=obsY(2,nse_day);
        store_recession{3,12}(nse_day,1)=obsY(3,nse_day);
        store_recession{4,12}(nse_day,1)=obsY(4,nse_day);
        store_recession{5,12}(nse_day,1)=obsY(5,nse_day);
        store_recession{6,12}(nse_day,1)=obsY(6,nse_day);
        store_recession{7,12}(nse_day,1)=obsY(7,nse_day);
        store_recession{8,12}(nse_day,1)=obsY(8,nse_day);
        store_recession{9,12}(nse_day,1)=obsY(9,nse_day);
        store_recession{10,12}(nse_day,1)=obsY(10,nse_day);
        store_recession{1,13}(nse_day,1)=testY(1,nse_day);
        store_recession{2,13}(nse_day,1)=testY(2,nse_day);
        store_recession{3,13}(nse_day,1)=testY(3,nse_day);
        store_recession{4,13}(nse_day,1)=testY(4,nse_day);
        store_recession{5,13}(nse_day,1)=testY(5,nse_day);
        store_recession{6,13}(nse_day,1)=testY(6,nse_day);
        store_recession{7,13}(nse_day,1)=testY(7,nse_day);
        store_recession{8,13}(nse_day,1)=testY(8,nse_day);
        store_recession{9,13}(nse_day,1)=testY(9,nse_day);
        store_recession{10,13}(nse_day,1)=testY(10,nse_day);
    end
    for nse_day=1:10
        NSE(nse_day)=(1-sum((store_recession{nse_day,12}-store_recession{nse_day,13}).^2)/(sum((store_recession{nse_day,12}-mean(store_recession{nse_day,12})).^2)));
    end
    %% Calculating NSE for all recession days combined
    for fn=1:size(obsY,2)
        recflowobs_all(1:10,fn)=obsY(1:10,fn);
        recflowpred_all(1:10,fn)=testY(1:10,fn);
    end
    recflowobs_all=recflowobs_all(:);
    recflowpred_all= recflowpred_all(:);
    NSE_all=(1-sum((recflowobs_all-recflowpred_all).^2)/(sum((recflowobs_all-mean(recflowobs_all)).^2)));
    ANNsall_120(i,1)=NSE_all;%%
    ANNs_120(i,1:10)=NSE;%%
    NSE_all=[];
    recflowobs_all=[];
    recflowpred_all=[];
    data_input=[];
    NSE=[];
    testX = [];
    testY = [];
    obsY =[];
    data_output=[];
    Data=[];
    flow=[];
    day=[];
    month= [];
    year=[];
    rec_imp=[];
    date_imp=[];
    Q_zero=[];
    LW=[];
    IW=[];
    b1=[];
    b2=[];
end
