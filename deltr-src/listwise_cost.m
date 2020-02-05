function J = listwise_cost(GAMMA, y, z, list_id, prot_idx)
    global CORES
    global DEBUG
    global ONLY_L
    global ONLY_U
    global L_AND_U
    
    ly = @(i) y(find(list_id == list_id(i)),:);
    lz = @(i) z(find(list_id == list_id(i)),:);
    
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
    % exposure_difference shall be zero when the protected group already has more 
    % exposure then the non-protected, such protected are not ranked down
    % max also makes sure exposure is not NaN, but 0 instead 
    % can be NaN if either protected or non-protected group has size 0
    % !!BECAUSE WE WANT TO PREDICT LOWER Z FOR BETTER RANKS
    exposure_diff = @(i) (max(0, (exposure_nprot_normalized(i) - exposure_prot_normalized(i))))^2;
    
    % calculate accuracy wrt training data
    accuracy = @(i) (-sum(topp(ly(i)) .* log( topp(lz(i)) )));
        
    if DEBUG
      iter = 1;
      idx = prot_idx_per_query(iter);
      z_prot = l_prot_vec(lz(iter), prot_idx_per_query(iter));
      z_nprot = l_prot_vec(lz(iter), !prot_idx_per_query(iter));

      top1_prot = topp_prot(l_prot_vec(lz(iter), prot_idx_per_query(iter)), lz(iter));
      top1_nprot = topp_prot(l_prot_vec(lz(iter), !prot_idx_per_query(iter)), lz(iter));
      top1_prot_times_v = topp_prot(l_prot_vec(lz(iter), prot_idx_per_query(iter)), lz(iter)) ./ log(2);
      top1_nprot_times_v = topp_prot(l_prot_vec(lz(iter), !prot_idx_per_query(iter)), lz(iter)) ./ log(2);
      
      group_size_prot = group_size_p(iter);
      group_size_nprot = group_size_np(iter);
      
      exposure_p = exposure_prot(iter);
      exposure_p_norm = exposure_prot_normalized(iter);
      
      exposure_np = exposure_nprot(iter);
      exposure_np_norm = exposure_nprot_normalized(iter);
      
      exposure_difference = exposure_diff(iter);
      accuracy2 = accuracy(iter);
      
      cost = GAMMA * exposure_diff(iter) .+ accuracy(iter);
    end
    
    if ONLY_U
      j = @(i) GAMMA * exposure_diff(i);
    end
    
    if ONLY_L
      j = @(i) accuracy(i);
    end
    
    if L_AND_U
      %u = @(i) GAMMA * exposure_diff(i);
      %l = @(i) accuracy(i);
      j = @(i) GAMMA * exposure_diff(i) .+ accuracy(i);
    end 
    J = pararrayfun(CORES, j,1:size(z,1), "VerboseLevel", 0);
end
