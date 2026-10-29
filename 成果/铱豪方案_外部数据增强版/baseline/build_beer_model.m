function m = build_beer_model(inputDir)
% R2021a+; base MATLAB only. Data columns retain their original spelling.
T = readtable(fullfile(inputDir,'1.csv'),'VariableNamingRule','preserve');
S = readtable(fullfile(inputDir,'4.csv'),'VariableNamingRule','preserve');
assert(isequal(T.beer,S.beer),'Beer row keys differ.');
assert(numel(unique(T.beer))==height(T),'Duplicate beer keys.');
assert(isequal(string(T.tasting_category_fine),string(S.tasting_category_fine)));
X = T{:,4:end}; Y = S{:,4:end};
assert(all(isfinite(Y),'all'),'Sensory data must be finite.');
m.features = T.Properties.VariableNames(4:end);
m.sensory = S.Properties.VariableNames(4:end);
fold = readmatrix(fullfile(inputDir,'folds.csv'));
assert(numel(fold)==height(T) && all(ismember(fold,1:5)));
pred = zeros(size(Y));
for k=1:5
    tr=fold~=k; va=~tr;
    p=fitRidge(X(tr,:),Y(tr,:));
    pred(va,:)=predictRidge(X(va,:),p);
end
m.cvR2=1-sum((pred-Y).^2,1)./sum((Y-mean(Y,1)).^2,1);
m.cvMAE=mean(abs(pred-Y),1);
m.ridge=fitRidge(X,Y);
Xi=impute(X,m.ridge.median);
srcRows=find(string(T.tasting_category_fine)=="Pils/Lager");
tgtRows=find(string(T.tasting_category_fine)=="Tripel");
assert(~isempty(srcRows)&&~isempty(tgtRows),'Missing source/target styles.');
srcData=Xi(srcRows,:);
[~,k]=min(mean(((srcData-mean(srcData,1))./m.ridge.sd).^2,2));
m.sourceRow=srcRows(k); m.sourceBeer=T.beer(m.sourceRow);
m.source=Xi(m.sourceRow,:);
m.target=median(Xi(tgtRows,:),1);
m.targetY=mean(Y(tgtRows,:),1);
m.yScale=std(Y,0,1); m.yScale(m.yScale<1e-12)=1;
m.defectNames={'DMS','X2.3.butanedione','X4VG','X.E..2.nonenal'};
[found,m.di]=ismember(m.defectNames,m.features); assert(all(found));
m.logDefect=[true false true false];
% Pin transformations to the supplied calibration evidence, not name heuristics.
V=readtable(fullfile(inputDir,'00_数值变换核验.csv'),'VariableNamingRule','preserve');
[found,rows]=ismember(m.defectNames,string(V.compound)); assert(all(found));
assert(isequal(reshape(string(V.transform(rows))=="log10",1,[]),m.logDefect));
m.sourceConcentration=m.source(m.di);
m.sourceConcentration(m.logDefect)=10.^m.sourceConcentration(m.logDefect);
assert(all(m.sourceConcentration>0));
m.threshold=[17.25 .29925 .206333333 Inf];
m.cap=.95*min(m.sourceConcentration,m.threshold);
m.stale=find(strcmp(m.sensory,'stale_hops')); assert(isscalar(m.stale));
m.sourceY=predictRidge(m.source,m.ridge);
% Third knob: the strongest attainable stale_hops decrease under this surrogate.
lo=quantileLinear(Xi,.01); hi=quantileLinear(Xi,.99);
bound=hi; bound(m.ridge.B(:,m.stale)>=0)=lo(m.ridge.B(:,m.stale)>=0);
benefit=(m.source-bound)./m.ridge.sd.*m.ridge.B(:,m.stale)';
benefit(m.di)=-Inf;
benefit(endsWith(string(m.features),'_sum'))=-Inf;
[~,m.control]=max(benefit); m.controlBound=bound(m.control);
m.sourceCommit='3e41cdf5380ea05ba9c37acbdd4992a062fc8e14';
end

function p=fitRidge(X,Y)
p.median=median(X,1,'omitnan'); assert(all(isfinite(p.median)));
X=impute(X,p.median);
p.lo=quantileLinear(X,.01); p.hi=quantileLinear(X,.99);
X=min(max(X,p.lo),p.hi);
p.mu=mean(X,1); p.sd=std(X,0,1); p.sd(p.sd<1e-12)=1;
Z=(X-p.mu)./p.sd; p.yMean=mean(Y,1); p.lambda=1000;
p.B=(Z'*Z+p.lambda*eye(size(Z,2)))\(Z'*(Y-p.yMean));
end

function Y=predictRidge(X,p)
X=impute(X,p.median); X=min(max(X,p.lo),p.hi);
Y=((X-p.mu)./p.sd)*p.B+p.yMean;
end

function X=impute(X,med)
for j=1:size(X,2)
    X(~isfinite(X(:,j)),j)=med(j);
end
end

function q=quantileLinear(X,p)
% NumPy type-7 quantile parity; MATLAB default quantile convention can differ.
X=sort(X,1); t=1+(size(X,1)-1)*p; a=floor(t); b=ceil(t);
q=X(a,:)+(t-a)*(X(b,:)-X(a,:));
end
