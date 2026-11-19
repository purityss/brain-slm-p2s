function r = slm_plot_safe_corr(x,y,kind)
% Pearson or Spearman correlation using base MATLAB only.
% Pairwise non-finite values are dropped; tied ranks use their mean rank.
x=double(x(:)); y=double(y(:));
if numel(x)~=numel(y), error('Correlation inputs must have the same length.'); end
good=isfinite(x)&isfinite(y); x=x(good); y=y(good);
if numel(x)<3 || std(x)==0 || std(y)==0, r=nan; return; end
if strcmpi(kind,'Spearman')
    x=average_tied_ranks(x); y=average_tied_ranks(y);
elseif ~strcmpi(kind,'Pearson')
    error('Unsupported correlation type: %s',kind);
end
x=x-mean(x); y=y-mean(y);
denominator=norm(x)*norm(y);
if denominator==0 || ~isfinite(denominator), r=nan; return; end
r=max(-1,min(1,(x'*y)/denominator));
end

function ranks = average_tied_ranks(x)
[sorted,order]=sort(x(:));
boundaries=[1;find(diff(sorted)~=0)+1;numel(sorted)+1];
ranks=zeros(size(sorted));
for i=1:numel(boundaries)-1
    first=boundaries(i); last=boundaries(i+1)-1;
    ranks(order(first:last))=(first+last)/2;
end
end
