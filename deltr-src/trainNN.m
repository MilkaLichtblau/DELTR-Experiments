function [omega, avg_J] = trainNN(GAMMA, directory, list_id, X, y, T, e, quiet=true)
    % load constants
    source globals.m;

    m = size(X,1);
    n_features = size(X,2);
    n_lists = size(unique(list_id),1);
    
    prot_idx = ( X(:,PROT_COL)==PROT_ATTR ); 
    
    % linear neural network parameter initialization
    omega = rand(n_features,1)*INIT_VAR;
    %omega = [0.0069304; 0.0084614];
    
    cost_converge_J = zeros(T, 1);
    cost_converge_L = zeros(T, 1);
    cost_converge_U = zeros(T, 1);
    omega_converge = zeros(T, n_features);

    for t = 1:T
      fprintf(".")

        if quiet == false
            fprintf("iteration %d: ", t)
        end
        
        % forward propagation
        z =  X * omega;
        
        % cost
        if quiet == false
            fprintf("computing cost... \n")
        end

        % with regularization
        cost = listwise_cost(GAMMA, y,z, list_id, prot_idx);
        %fprintf("cost=%f\n", sum(cost));
        %cost_converge(t) = sum(cost);
        %fprintf("cost: %f\n", J(1));
        J = cost + ((z.*z)'.*LAMBDA);
        cost_converge_J(t) = sum(J);

        % without regularization
        %J = listwise_cost(y,z, list_id, prot_idx);
        
        % gradient
        if quiet == false
            fprintf("computing gradient...\n")
        end

        grad = listnet_gradient(GAMMA, X, y, z, list_id, prot_idx);
        %fprintf("grad: %f\n", grad(1));

        % parameter update
        omega = omega - (e .* sum(grad',2));
        %fprintf("omega: %f\n", omega);
        omega_converge(t, :) = omega(:);

        
        if quiet == false
            fprintf("\n")
        end
    end
    fprintf("\n")

    cost_filename = [directory "cost.png"];
    gradient_filename = [directory "gradient.png"];
    hf = figure('visible','off'); plot(cost_converge_J); print(hf, cost_filename, '-dpng');
    gf = figure('visible','off'); plot(omega_converge); print(gf, gradient_filename, '-dpng');
end

