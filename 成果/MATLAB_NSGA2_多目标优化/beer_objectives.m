function [F,c,V,Yhat] = beer_objectives(U,m)
% U rows = [alpha beta gamma], each in [0,1]; all three objectives minimized.
assert(size(U,2)==3 && all(isfinite(U),'all'));
V=m.source+U(:,1).*(m.target-m.source);
C=V(:,m.di); C(:,m.logDefect)=10.^C(:,m.logDefect);
C=min(C,m.cap).*2.^(-U(:,2));
V(:,m.di)=C;
V(:,m.di(m.logDefect))=log10(C(:,m.logDefect));
V(:,m.control)=V(:,m.control)+U(:,3).*(m.controlBound-V(:,m.control));
% Evaluate the final corrected profile, never the uncorrected candidate.
Z=(min(max(V,m.ridge.lo),m.ridge.hi)-m.ridge.mu)./m.ridge.sd;
Yhat=Z*m.ridge.B+m.ridge.yMean;
F=[mean(((Yhat-m.targetY)./m.yScale).^2,2), ...
   sqrt(mean(((V-m.source)./m.ridge.sd).^2,2)), mean(C./m.cap,2)];
c=(Yhat(:,m.stale)-m.sourceY(m.stale))/m.yScale(m.stale);
assert(all(isfinite(F),'all') && all(isfinite(c)));
end
