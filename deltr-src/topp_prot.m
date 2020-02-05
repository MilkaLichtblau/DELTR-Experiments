% topp_prot.m
% 
% This function computes the top-one probabilities of the elements of a 
% a given vector u that should contain only elements of one particular 
% (protected) group

function t_prot = topp_prot(u, v)
  t_prot = exp(u)/sum(exp(v));
end