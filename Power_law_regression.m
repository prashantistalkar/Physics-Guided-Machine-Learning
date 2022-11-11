%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Written by Akshay Kadu
%Email: akshaykadu5626@gmail.com
%Indian Institute of Technology Bombay, Mumbai, India
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all  % clearing all previus variables
files=dir('*.csv') ;  % reading all csv files in the folder 
%% running the code for file no 
i=1  ;  % running code file no 1 in the folder 
%%  Runing the code 
Data=xlsread(files(i).name);
flow=Data(:,4) ;
Date=Data(:,1:3) ;

[m n]=size(flow); %Determining the size of flow variable
%Initializing few variables
rec_imp= [];
date_imp=[];
store_recession= cell(1,1);
Q_zero=cell(1,1);
j = 1;

%% Sorting receseeion events from the flow time-series
for i=123:m-1
    if flow(i+1)<flow(i)
        rec_imp1=flow(i);
        date_imp1=Date(i);
        rec_imp= vertcat(rec_imp,rec_imp1);
        date_imp= vertcat(date_imp,date_imp1);
    else
        rec_imp1=flow(i);
        rec_imp= vertcat(rec_imp,rec_imp1);
        date_imp1=Date(i);
        date_imp= vertcat(date_imp,date_imp1);  
        if (length(rec_imp)>10) % Sorting out the recession events longer than 10 days and storing them in cell
            Q_zero{j,1}=rec_imp(1); %Storing peak discharges in separate cell
            Q_zero{j,2}=date_imp(1);
            store_recession{j,1}= rec_imp(2:end);
            store_recession{j,2}= date_imp(2:end);
            
            in_10=(i-length(rec_imp)-10);%Previous 10th day starting from 2 days before peak
            in_30=(i-length(rec_imp)-30);
            in_60=(i-length(rec_imp)-60);
            in_120=(i-length(rec_imp)-120);
            ed= (i-length(rec_imp)-1); % end date (2 days before peak)
            
            flow_prev_10=flow(in_10:(ed));
            flow_prev_30=flow(in_30:(ed));
            flow_prev_60=flow(in_60:(ed));
            flow_prev_120=flow(in_120:(ed));
            
            mQ_10=mean(flow_prev_10); %Calculating Mean past discharge
            mQ_30=mean(flow_prev_30);
            mQ_60=mean(flow_prev_60);
            mQ_120=mean(flow_prev_120);
            
            if mQ_10< mQ_30             % Applying (Varaprasad et al., 2020) methodology
                Qn_avg=mQ_10;
            elseif(mQ_30<mQ_60)
                Qn_avg=mQ_30;
            elseif(mQ_60<mQ_120)
                Qn_avg=mQ_60;
            else
                Qn_avg=mQ_120;
            end
            store_recession{j,3}=Qn_avg; %Storing final required past mean discharge
            j = j+1;
            rec_imp =[];
            date_imp= [];
        end
        rec_imp =[];
        date_imp= [];
    end
end

%% Last recession event
if (length(rec_imp)>10) % Sorting out the recession events longer than 10 days and storing them in cell
    Q_zero{j,1}=rec_imp(1); %Storing peak discharges in separate cell
    Q_zero{j,2}=date_imp(1);
    store_recession{j,1}= rec_imp(2:end);
    store_recession{j,2}= date_imp(2:end);
    
    in_10=(i-length(rec_imp)-10);%Previous 10th day starting from 2 days before peak
    in_30=(i-length(rec_imp)-30);
    in_60=(i-length(rec_imp)-60);
    in_120=(i-length(rec_imp)-120);
    ed= (i-length(rec_imp)-1); % end date (2 days before peak)
    
    flow_prev_10=flow(in_10:(ed));
    flow_prev_30=flow(in_30:(ed));
    flow_prev_60=flow(in_60:(ed));
    flow_prev_120=flow(in_120:(ed));
    
    mQ_10=mean(flow_prev_10); %Calculating Mean past discharge
    mQ_30=mean(flow_prev_30);
    mQ_60=mean(flow_prev_60);
    mQ_120=mean(flow_prev_120);
    
    if mQ_10< mQ_30             % Applying (Varaprasad et al., 2020) methodology
        Qn_avg=mQ_10;
    elseif(mQ_30<mQ_60)
        Qn_avg=mQ_30;
    elseif(mQ_60<mQ_120)
        Qn_avg=mQ_60;
    else
        Qn_avg=mQ_120;
    end 
    store_recession{j,3}=Qn_avg; %Storing the final required past mean discharge (QN)
    j = j+1;
    rec_imp =[];
    date_imp= [];  
end
%% Training and Testing
[o,v] = size(store_recession) ;
%Partioning the recession events for training and testing ;
Training = store_recession(1:round(0.6*o),:) ; % Using 60% of recession events for model training
Testing = store_recession((round(0.6*o)+1:end),:) ;
Q_zero_testing=Q_zero((round(0.6*o)+1:end),:);
train_eve=size(Training,1);
%% Calculating dQ/dt and Q in Power-law equation
for a=1:train_eve
    rec_flow= Training{a,1};
    c= length(rec_flow);
    for ii=2:c
        delta_q(ii-1,1)= (rec_flow(ii-1)-rec_flow(ii));% dQ/dt
        avg_q(ii-1,1)= (rec_flow(ii)+rec_flow(ii-1))/2;% Q
    end
    %% Estimating parameters alpha and k using Least Square Linear Regression
    log_deltaq= log(delta_q);
    log_avgq= log(avg_q);
    para= polyfit(log_avgq,log_deltaq,1);
    alpha(a)=para(1);
    storage_coeff(a)=para(2);
    storage_coeff(a)= exp(storage_coeff(a));
    rec_flow=[];
    delta_q=[];
    avg_q=[];
    log_deltaq=[];
    log_avgq=[];
    para=[];
    storage_coeff=[];
end
medn_alpha= median(alpha); %Calculating characteristic median alpha for basin
%% Recalculating parameter k by fixing alpha at median value
for d=1:train_eve   
    rec_flow= Training{d,1};
    e= length(Training{d,1});
    for j=2:e
        delta_q(j-1,1)= (rec_flow(j-1)-rec_flow(j));%dQ/dt
        avg_q(j-1,1)= (rec_flow(j)+rec_flow(j-1))/2;%Q
        q_power_alpha(j-1,1)= avg_q(j-1,1)^medn_alpha;
    end
    event_storg_coeff(d,1)=q_power_alpha\delta_q; %k for each recession event keeping alpha fixed to its median value
    Training{d,4}=event_storg_coeff(d);
    rec_flow=[];
    delta_q=[];
    avg_q=[];
    q_power_alpha=[];
end
for p_avg=1:train_eve
    avg_Q(p_avg,1)= Training{p_avg,3};
end
%% Least square linear regression between k and Q_avg to estimate k' and lymbda
basin_para=polyfit(log(avg_Q),log(event_storg_coeff),1);
lymbda_avg=basin_para(1);
k_avg= exp(basin_para(2));
% End of model training
%% Recession flow prediction and model performance evaluation
for tst=1:size(Testing,1)
    avg_Q_test(tst)= Testing{tst,3};
    storage_coeff_predict(tst)= k_avg.*((avg_Q_test(tst)).^lymbda_avg); %Predicting k
    Testing{tst,4}=storage_coeff_predict(tst);
    Q_rec_zero= Q_zero_testing{tst,1}(1);
    for te=1:length(Testing{tst,1})   %Prediction of recession flow using power-law model
        rec_Q(te)=real(Q_rec_zero*(1+(medn_alpha-1)*storage_coeff_predict(tst)*te*(Q_rec_zero).^(medn_alpha-1)).^(1/(1-medn_alpha)));
        rec_Q(te)= max(rec_Q(te),0);
    end
    Testing{tst,5}= rec_Q';
    rec_Q=[];
end
%% Plotting sample observed and predicted recession event
plot(Testing{1,1},'b')% first observed event
hold on
plot(Testing{1,5},'r')% first predicted event
xlabel('Recession Days')
ylabel('Discharge (cfs)')
legend('Qobs','Qpred')

%% Overall NSE
recflowobs_all=[];
recflowpred_all=[];
for fn=1:size(Testing,1)
    recflowobs_all1(:,1)=Testing{fn,1}(:,1);
    recflowobs_all=vertcat(recflowobs_all,recflowobs_all1);
    recflowobs_all1=[];
    recflowpred_all1(:,1)=Testing{fn,5}(:,1);
    recflowpred_all=vertcat(recflowpred_all, recflowpred_all1);
    recflowpred_all1=[];
end
NSE_all=(1-sum((recflowobs_all-recflowpred_all).^2)/(sum((recflowobs_all-mean(recflowobs_all)).^2)))
NSE_log__all=(1-sum((log(recflowobs_all)-log(recflowpred_all)).^2)/(sum((log(recflowobs_all)-mean(log(recflowobs_all))).^2)))



