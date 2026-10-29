function mask = nondominated_mask(F)
% No toolbox needed. Strict dominance; equal rows do not dominate each other.
assert(all(isfinite(F),'all'));
mask=true(size(F,1),1);
for i=1:size(F,1)
    mask(i)=~any(all(F<=F(i,:),2)&any(F<F(i,:),2));
end
end
