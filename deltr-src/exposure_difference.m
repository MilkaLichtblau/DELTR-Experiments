function exposure_diff = exposure_difference()
    % get idx of protected candidates per query, otherwise dimensions don't fit
    prot_idx_per_query = @(i) prot_idx(find(list_id == list_id(i)),:);
    
    l_prot_vec = @(preds, idx) preds(idx);
    
    group_size_p = @(i) size(l_prot_vec(lz(i), prot_idx_per_query(i)), 1);
    group_size_np = @(i) size(l_prot_vec(lz(i), !prot_idx_per_query(i)), 1);
 
    % Exposure in Rankings for the protected and non-protected group
    exposure_prot = @(i) sum(topp_prot(l_prot_vec(lz(i), prot_idx_per_query(i)), lz(i)) ./ log(2)); 
    exposure_prot_normalized = @(i) exposure_prot(i) / group_size_p(i); 
    
    exposure_nprot = @(i) sum(topp_prot(l_prot_vec(lz(i), !prot_idx_per_query(i)), lz(i)) ./ log(2));
    exposure_nprot_normalized = @(i) exposure_nprot(i) / group_size_np(i); 
   
    % calculate difference of exposure between the two groups
    exposure_diff = @(i) (exposure_prot_normalized(i) - exposure_nprot_normalized(i))^2;
    
    % make sure exposure is not NaN, but 0 instead 
    % can be NaN if either protected or non-protected group has size 0
    exposure_non_nan = @(i) max(exposure_diff(i), 0);    
end