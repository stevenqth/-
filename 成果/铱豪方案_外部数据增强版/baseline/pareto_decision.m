function [D,Z] = pareto_decision(U,F)
% Weighted TOPSIS after nondominated filtering. Weights are preferences.
% sqrt(weight)*normalized distance makes weights linear in squared distance.
span=max(F,[],1)-min(F,[],1); active=span>1e-12;
Z=zeros(size(F)); Z(:,active)=(F(:,active)-min(F(:,active),[],1))./span(active);
W=[1 1 1; .6 .2 .2; .2 .6 .2; .2 .2 .6];
names=["balanced";"sensory_first";"small_change";"low_defect"];
idx=zeros(4,1); score=zeros(4,1);
for k=1:4
    w=W(k,:).*active;
    if sum(w)==0
        closeness=.5*ones(size(F,1),1);
    else
        w=w/sum(w);
        dp=sqrt(sum(Z.^2.*w,2));
        dm=sqrt(sum((Z-double(active)).^2.*w,2));
        closeness=.5*ones(size(dp)); valid=dp+dm>0;
        closeness(valid)=dm(valid)./(dp(valid)+dm(valid));
    end
    [score(k),idx(k)]=max(closeness);
end
D=table(names,idx,U(idx,1),U(idx,2),U(idx,3),F(idx,1),F(idx,2),F(idx,3),score, ...
    'VariableNames',{'scenario','front_id','alpha','beta','gamma','f1','f2','f3','topsis'});
end
