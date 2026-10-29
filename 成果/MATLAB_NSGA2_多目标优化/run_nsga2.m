function result = run_nsga2()
% Run in MATLAB R2021a+ with Global Optimization Toolbox.
% >> addpath('absolute/path/to/this/folder'); result=run_nsga2;
% gamultiobj is MathWorks' controlled elitist NSGA-II variant, not Deb's exact code.
here=fileparts(mfilename('fullpath')); out=fullfile(here,'matlab_results');
assert(exist('gamultiobj','file')==2, ...
    'Global Optimization Toolbox (gamultiobj) is required.');
if ~exist(out,'dir'), mkdir(out); end
m=build_beer_model(fullfile(here,'inputs'));
% Numerical parity check against independently computed Python fixtures.
fixtures=readmatrix(fullfile(here,'inputs','objective_fixtures.csv'));
[f,c]=beer_objectives(fixtures(:,1:3),m);
err=max(abs([f,c]-fixtures(:,4:7)),[],'all');
assert(err<1e-8,'Python/MATLAB objective parity failed: %.3g',err);
assert(isequal(nondominated_mask([1 2;2 1;3 3;1 2]),logical([1;1;0;1])));
[dc,~]=pareto_decision([0 0 0;1 1 1],[1 1 1;1 1 1]);
assert(all(dc.topsis==.5),'Constant-objective TOPSIS check failed.');

seeds=[42 43 44]; allU=[]; allF=[]; runs=struct([]);
for r=1:numel(seeds)
    rng(seeds(r),'twister');
    % Feasible initialization respects stale_hops; GA still checks every offspring.
    pool=rand(3000,3); [~,c]=beer_objectives(pool,m);
    pool=pool(c<=1e-8,:);
    assert(size(pool,1)>=160,'Not enough feasible initial points; inspect model.');
    opts=optimoptions('gamultiobj','PopulationSize',160, ...
        'MaxGenerations',200,'MaxStallGenerations',60, ...
        'ParetoFraction',.7,'FunctionTolerance',1e-6, ...
        'ConstraintTolerance',1e-8,'UseParallel',false, ...
        'InitialPopulationMatrix',pool(1:160,:),'Display','iter');
    started=tic;
    [u,f,flag,info]=gamultiobj(@(u)objective(u,m),3,[],[],[],[], ...
        zeros(1,3),ones(1,3),@(u)constraints(u,m),opts);
    elapsed=toc(started);
    % Recompute objectives and constraints after solving; reject infeasible output.
    [f,c,v]=beer_objectives(u,m);
    conc=v(:,m.di); conc(:,m.logDefect)=10.^conc(:,m.logDefect);
    ok=c<=1e-8 & all(u>=-1e-10 & u<=1+1e-10,2) ...
        & all(conc<=m.cap.*(1+1e-10),2);
    allU=[allU;u(ok,:)]; allF=[allF;f(ok,:)]; %#ok<AGROW>
    runs(r).seed=seeds(r); runs(r).exitflag=flag;
    runs(r).output=info; runs(r).seconds=elapsed;
    runs(r).returned=size(u,1); runs(r).feasible=sum(ok);
    writetable(array2table([u,f,c],'VariableNames', ...
        {'alpha','beta','gamma','f1','f2','f3','stale_constraint'}), ...
        fullfile(out,sprintf('seed_%d.csv',seeds(r))));
end
assert(~isempty(allU),'No feasible solution; no recommendation can be made.');
[allU,ia]=unique(allU,'rows','stable'); allF=allF(ia,:);
keep=nondominated_mask(allF); U=allU(keep,:); F=allF(keep,:);
[D,Z]=pareto_decision(U,F); [~,c,V,Yhat]=beer_objectives(U,m);
assert(all(c<=1e-8) && all(nondominated_mask(F)));
front=array2table([U,F,c],'VariableNames', ...
    {'alpha','beta','gamma','f1','f2','f3','stale_constraint'});
front=addvars(front,(1:height(front))','Before',1,'NewVariableNames','front_id');
writetable(front,fullfile(out,'pareto_nsga2.csv'));
writetable(D,fullfile(out,'decisions_nsga2.csv'));
writetable(array2table(V(D.front_id,:),'VariableNames',m.features), ...
    fullfile(out,'recommended_chemical_profiles.csv'));
writetable(array2table(Yhat(D.front_id,:),'VariableNames',m.sensory), ...
    fullfile(out,'recommended_sensory_predictions.csv'));
writetable(table(string(m.sensory)',m.cvR2',m.cvMAE', ...
    'VariableNames',{'target','exploratory_oof_R2','exploratory_oof_MAE'}), ...
    fullfile(out,'surrogate_validation.csv'));

fig=figure('Color','w');
scatter3(F(:,1),F(:,2),F(:,3),22,U(:,1),'filled'); hold on;
scatter3(D.f1,D.f2,D.f3,90,'kp','filled');
xlabel('f1: target sensory error'); ylabel('f2: chemical adjustment');
zlabel('f3: relative chemical defect burden');
title('MATLAB NSGA-II variant: approximate Pareto set');
cb=colorbar; cb.Label.String='alpha'; grid on;
exportgraphics(fig,fullfile(out,'pareto_3d.png'),'Resolution',200);
fig=figure('Color','w');
plot(1:3,Z(D.front_id,:)','-o','LineWidth',1.5);
xticks(1:3); xticklabels({'Sensory error','Adjustment','Defect burden'});
ylabel('Normalized objective (lower is better)');
legend(D.scenario,'Interpreter','none','Location','best'); grid on;
exportgraphics(fig,fullfile(out,'decision_tradeoffs.png'),'Resolution',200);

result.sourceCommit=m.sourceCommit; result.matlabVersion=version;
result.engine='MATLAB gamultiobj controlled elitist NSGA-II variant';
result.runs=runs; result.objectiveParityMaxError=err;
result.sourceBeer=m.sourceBeer; result.sourceRow=m.sourceRow;
result.staleControlFeature=m.features{m.control};
result.cvMeanR2=mean(m.cvR2); result.cvStaleR2=m.cvR2(m.stale);
result.nPareto=size(F,1); result.maxStaleConstraint=max(c);
result.inferenceScope='In-silico exploratory study; no sensory or brewing validation';
% Compare to a finite reference grid, without claiming either is globally optimal.
G=readmatrix(fullfile(here,'reference_grid','pareto_grid.csv'));
GF=G(:,5:7); isDominated=false(size(F,1),1);
for i=1:size(F,1)
    isDominated(i)=any(all(GF<=F(i,:)+1e-10,2)&any(GF<F(i,:)-1e-10,2));
end
result.fractionDominatedByReferenceGrid=mean(isDominated);
fid=fopen(fullfile(out,'run_report.json'),'w','n','UTF-8');
assert(fid~=-1); cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'%s',jsonencode(result,'PrettyPrint',true));
save(fullfile(out,'run_workspace.mat'),'m','U','F','D','result');
disp(D); fprintf('Saved results to %s\n',out);
end

function f=objective(u,m)
f=beer_objectives(reshape(u,1,[]),m);
end

function [c,ceq]=constraints(u,m)
[~,c]=beer_objectives(reshape(u,1,[]),m); ceq=[];
end
