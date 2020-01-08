function Metrics = computeMetric(referData,predData)
%

Metrics=zeros(size(predData,3),4);
for i=1:size(predData,3)
    refer=double(referData(:,:,i));
    pred=double(predData(:,:,i));
    Metrics(i,4)=ssim(refer,pred);
    refer=refer(:);
    pred=pred(:);    
    Metrics(i,1)=corr(refer,pred).^2;% R2
    Metrics(i,2)=sqrt(mse(refer,pred)); % RMSE
    Metrics(i,3)=1 -  sum( (refer - pred).^2) ./ ...
        sum( ( abs(pred - mean(refer)) + abs(refer - mean(refer)) ).^2 ); % IA
end
Metrics=Metrics';
Metrics=round(Metrics(:),4);
end
