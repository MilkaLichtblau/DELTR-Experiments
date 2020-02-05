function [omega, avg_J] = trainNN(list_id, directory, X, y, T, e, quiet=true)
    % load constants
    source globals.m;

    m = size(X,1);
    n_features = size(X,2);
    n_lists = size(unique(list_id),1);

    % linear neural network parameter initialization
    omega = rand(n_features,1)*INIT_VAR;
    %omega = [0.009; 0.001];
    
    cost_converge = zeros(T, 1);
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
            fprintf("computing cost... ")
        end

        % with regularization
        J = listwise_cost(y,z, list_id);
        cost = J  + ((z.*z)'.*LAMBDA);
        %fprintf("cost %f\n", J);
        % without regularization
        %J = listwise_cost(y,z, list_id);
        cost_converge(t) = sum(J);

        
        % gradient
        if quiet == false
            fprintf("computing gradient...")
        end

        grad = listnet_gradient(X, y, z, list_id);
        %fprintf("z %f\n", z(:, 1));
        % parameter update
        omega = omega - (e .* sum(grad',2));
        omega_converge(t, :) = omega;
        
        if quiet == false
            fprintf("\n")
        end
    end
    fprintf("\n")

    %cost_filename = [directory "cost.png"];
    %gradient_filename = [directory "gradient.png"];
    %hf = figure('visible','off'); plot(cost_converge); print(hf, cost_filename, '-dpng');
    %gf = figure('visible','off'); plot(omega_converge); print(gf, gradient_filename, '-dpng');
end

